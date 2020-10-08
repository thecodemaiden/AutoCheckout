import queue
import numpy as np
import time
import math

from collections import defaultdict, Counter

import sys
import os
import logging
import pickle
import json

#from cpsdriver.clients import (
from clients import (
    CpsMongoClient,
    CpsApiClient,
    TestCaseClient,
)
from cli import parse_configs
from log import setup_logger

#from options import cpsdriver_args
#print (cpsdriver_args)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
setup_logger("info")

def moving_avg(a,n=10,axis=None):
    if n >= len(a):
        return np.mean(a, axis=axis)

    ret = np.cumsum(a, axis=axis)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:]/n

def main(args=None):
    args = parse_configs(args)
    #setup_logger(args.log_level)
    mongo_client = CpsMongoClient(args.db_address)
    api_client = CpsApiClient()
    test_client = TestCaseClient(mongo_client, api_client)
    #test_client.load(f"{args.command}-{args.sample}")
    logger.info(f"Available Test Cases are {test_client.available_test_cases}")
    test_client.set_context(args.command, load=False)
    shopping = create_shopping_list(test_client, args.command)
    generate_receipts(shopping, args.command, args.token)

def load_product_locations(test_client):
    productList = test_client.list_products()
    #out_sensor_product_info = []
    weight_sensor_info = defaultdict(list)
    product_lookup_info = {}
    for aProduct in productList:
        item_info = {'name':aProduct.name, 'barcode':aProduct.product_id.barcode, 'weight':aProduct.weight}
        allFacings = test_client.find_product_facings(aProduct.product_id)
        if aProduct.weight> 0 and len(allFacings) > 0:
            itemCoords = []
            for aFacing in allFacings:
                for plateLoc in aFacing.plate_ids:
                    location_id = (plateLoc.gondola_id, plateLoc.shelf_index, plateLoc.plate_index)
                    c = aFacing.coordinates
                    itemCoords.append( (c['dim_x'], c['dim_y'],c['dim_z']))
                    weight_sensor_info[location_id].append(aProduct.product_id)
            item_info['coords'] = itemCoords
            product_lookup_info[aProduct.product_id] = item_info
    return weight_sensor_info, product_lookup_info

def get_sensor_batch(test_client, start_time, batch_length):
    if start_time <= 0:
        # the first time, we don't know when the timestamps start, so let's find out
        first_data = test_client.find_first_after_time("plate_data",0.0)[0]
        start_time = first_data.timestamp

    batch_data = test_client.find_all_between_time("plate_data", start_time, start_time+batch_length)
    if len(batch_data) == 0:
        return None, -1
    weight_update_data = defaultdict(lambda: np.empty((0,2)))
    currentTime = start_time
    for rawData in batch_data:
        currentTime = rawData.timestamp
        startShelf = rawData.plate_id.shelf_index
        startPlate = rawData.plate_id.plate_index
        if startShelf > 0 or startPlate > 0:
            logger.warn("Data starts at shelf {}, plate {}".format(startShelf, startPlate))
        gondolaId = rawData.plate_id.gondola_id
        dataSize = rawData.data.shape
        nSamples = dataSize[0]
        nShelves = dataSize[1]
        nPlates = dataSize[2]
        ts = np.array(range(nSamples))*(1.0/60) + currentTime # the timestamps in this packet
        ts = ts.reshape((nSamples,1))
        for jj in range(nShelves):
            for kk in range(nPlates):
                weightData = (rawData.data[:,jj,kk]).reshape(nSamples,1)
                if not(np.isnan(weightData).all()):
                    sensor_id = (gondolaId, jj, kk)
                    updateData = np.hstack((ts,weightData))
                    prevData = weight_update_data[sensor_id]
                
                    weight_update_data[sensor_id] = np.vstack((prevData, updateData))

    return weight_update_data, currentTime            
    
def create_shopping_list(test_client, case_name):
    shoppingList = Counter()
    product_data_file = "../pickles/{}-products.pkl".format(case_name)
    if not os.path.exists(product_data_file):
        shelf_products, product_info = load_product_locations(test_client)
        with open(product_data_file, 'wb') as productOut:
            pickle.dump({'productsByPlate':shelf_products, 'productData':product_info}, productOut)
    else:
        with open(product_data_file, 'rb') as productIn:
            x = pickle.load(productIn)
            shelf_products = x['productsByPlate']
            product_info = x['productData']

    logger.info('Product data loaded')
    nextTime = -1
    while True:
        moreData, nextTime = get_sensor_batch(test_client, nextTime, 1.0)
        if moreData is None:
            break
        affected_plates = detect_plates_affected(moreData)
        #TODO: split these up by put back/pick up events, use shopping list to put back items
        #logger.debug("Event: {}".format(affected_plates))
        if len(affected_plates) > 0:
            basket_change = select_items_for_changes(affected_plates, shelf_products, product_info)
            if len(basket_change) > 0:
                shoppingList.update(basket_change)

    # just assign to the first target in the database
    targetList = test_client.find_first_after_time("full_targets",0.0)
    if targetList is None:
        targetList = test_client.find_first_after_time("targets",0.0)
    
    aTarget = targetList[0].targets[0]
    print("---")
    print(case_name)
    print('---')
    for (item, q) in shoppingList.items():
        name = product_info[item]['name']
        print("{}x {}".format(q,name))
    print("---")
    finalList = {aTarget.target_id:shoppingList}
    return finalList

def extract_product_ids(product_list):
    return set(product_list)

def select_items_for_changes(plate_changes, shelf_contents, product_info):
    # let's  first assume they all have the same timestamps, so no need to do this sort
    #XXX sortedChanges = sort(plate_changes, key=lambda c: c['time'][0])
    binnedChanges = []
    # let's put changes together if they are in adjacent plates carrying the same product
    sortedLocs = sorted(plate_changes.keys(), key=lambda l: 10000*l[0]+100*l[1]+l[2])
    lastLoc = sortedLocs[0]
    lastProductSet = extract_product_ids(shelf_contents[lastLoc])
    changeBin = {
        'candidates':lastProductSet, 'all_weights':[plate_changes[lastLoc]['weight']], 'all_plates':[lastLoc]}
    for ll in sortedLocs[1:]:
        makeNewBin = True
        if ll[0] == lastLoc[0] and ll[1] == lastLoc[1] and ll[2]-lastLoc[2] == 1:
            # on the same gondola and shelf, adjacent plates
            # do they share any products?
            thisProductSet = extract_product_ids(shelf_contents[ll]).union(lastProductSet)
            if len(thisProductSet) > 0:
                makeNewBin = False
                changeBin['candidates'] = thisProductSet
                lastProductSet = thisProductSet
                changeBin['all_weights'].append(plate_changes[ll]['weight'])
                changeBin['all_plates'].append(ll)
        lastLoc = ll
        if makeNewBin:
            # no product matched between them
            binnedChanges.append(changeBin)
            lastProductSet = extract_product_ids(shelf_contents[ll])
            plateWeight = plate_changes[ll]['weight']
            changeBin = {
                'candidates':lastProductSet, 'all_weights':[plateWeight], 'all_plates':[ll]}

    binnedChanges.append(changeBin)
    # now, let's examine those products to see what might have been picked
    item_change = defaultdict(int)
    for event in binnedChanges:
        # TODO: maybe prioritise multi-plate changes?
        chosenItem, quantity = findBestQuantityForItems(event, product_info)
        if chosenItem is not None:
            item_change[chosenItem] += quantity
    return item_change

def findBestQuantityForItems(event, product_info):
    nPlates = len(event['all_weights'])
    weight_percent_threshold = 1.0/3
    bestProduct = None
    leastVariance = np.inf
    productQuantity = 0
    # we're going to take 1,2,3,...n adjacents plates at a time
    for ii in range(nPlates): # how many plates to take -1
        for jj in range(nPlates-ii): # where to start taking from
            weightTotal = np.mean(event['all_weights'][jj:jj+ii+1])
            for prod in event['candidates']:
                info = product_info[prod]


                # look at total weight only for now
                possibleQuantity = math.floor(abs(weightTotal)/info['weight']+0.4)
                if possibleQuantity == 0 or possibleQuantity > 5: continue

                expectedWeight = possibleQuantity*info['weight']
                weightDiff = abs(weightTotal)-expectedWeight
                varRatio = weightDiff/info['weight']
                possibleQuantity = possibleQuantity*np.sign(weightTotal)
                logger.debug("{}x {} is off by {} ({})".format(possibleQuantity,info['name'], weightDiff, varRatio))
                if abs(varRatio) < weight_percent_threshold and abs(varRatio) < leastVariance:
                    bestProduct = prod
                    leastVariance = weightDiff/info['weight']
                    productQuantity = possibleQuantity
    return bestProduct, productQuantity


def detect_plates_affected(new_sensor_data):
    threshold = 5
    affectedPlates = {}
    for (sensorLoc, weightData) in new_sensor_data.items():
        # find the sum of the weight changes, if it's above the threshold, something happened on that plate
        startTime = weightData[0,0];
        endTime = weightData[-1,0];
        startWeight = weightData[0,1];
        endWeight = weightData[-1,1];
        filtWeight = np.array(moving_avg(weightData[:,1],10)).reshape(-1,1)
        #weightChange = np.mean(dWeight)
        #weightChange = startWeight-endWeight
        dW = filtWeight[0] - filtWeight[-1]
        if abs(dW) > threshold:
            weightChange = np.sign(dW)*(np.max(weightData[:,1])-np.min(weightData[:,1]))
            affectedPlates[sensorLoc] = {'time':(startTime,endTime), 'weight':weightChange}
    return affectedPlates

def generate_receipts(shopping_list, case_name, tokenStr):
    outfile = "../results/{}.json".format(case_name)
    uuid = get_uuid_for_case(case_name)
    submission = {"testcase":uuid, "user":tokenStr, "receipts":[]}
    for (shopper,basket) in shopping_list.items():
        bought = []
        for (item,quantity) in basket.items():
            if quantity <=0: continue
            bought.append({"barcode":item.barcode, "quantity":int(quantity)})
        submission["receipts"].append({"target_id":shopper, "products":bought})
    
    with open(outfile, 'w', encoding='utf8') as jsonOut:
        json.dump(submission, jsonOut)

def get_uuid_for_case(case_name):
    with open('../testcases.json','r') as jsonFile:
        data = json.load(jsonFile)
    for case in data:
        if case['name'] == case_name:
            return case['uuid']

if __name__ == "__main__":
    #files = os.listdir("/home/abannis/Research/AiFi/")
    #interesting = [f for f in files if "TEAM-" in f]
    #cases = [f.split('.')[0] for f in interesting]

#    for COMMAND in cases:
        #COMMAND="BASELINE-{}".format(ii+1)
    COMMAND="BASELINE-1"
    argstr = f"--command {COMMAND} --db-address {DB_ADDRESS} --api-address {API_ADDRESS} --token {TOKEN}"
    cpsdriver_args = argstr.split()
    main(cpsdriver_args)
