#!/usr/bin/env python
# coding: utf-8

# In[27]:


from flask import Flask, Response
import pickle as pkl
import logging
import pandas as pd
import sys

sys.path.append('/home/immichail/anaconda3/lib/python3.8/site-packages')

from fuzzysearch import find_near_matches


import flask
print('Flask version: ', flask.__version__)

from collections import Counter
import numpy as np
import logging
logging.getLogger().setLevel(logging.INFO)
import matplotlib.pyplot as plt
from flask import jsonify, make_response
from flask import request
import functools
import math

import random

from werkzeug.serving import run_simple

import pymongo

from dateutil.parser import parse as parseDate
from datetime import datetime, time, timedelta

from uuid import uuid4 as uuidf

def uuid():
    return str(uuidf())

db = pymongo.MongoClient()['ltcLongevity']

DAYS_DICT = {
    'Пн': 0,
    'Вт': 1,
    'Ср': 2,
    'Чт': 3,
    'Пт': 4,
    'Сб': 5,
    'Вс': 6
}

DATEFORMAT = '%d.%m.%Y'


# In[28]:


from math import sin, cos, sqrt, atan2, radians

defaultCoords = {'lat': 55.651577069999995, 'lon': 37.71133880666667}

def getUserCoords(userId: int):
    userCoords = db.usersV2.find_one({'userId': userId})

    if 'addressCoords' in userCoords:
        userCoords = userCoords['addressCoords']
    else:
        userCoords = defaultCoords
        
    return userCoords

def getGroupCoords(groupId: int):
    groupCoords = db.schedule.find_one({'groupId': groupId})

    if 'addressCoords' in userCoords:
        groupCoords = groupCoords['addressCoords']
    else:
        groupCoords = defaultCoords
        
    return groupCoords

def calcDistance(c1, c2):
    
    R = 6373.0

    lat1 = radians(c1['lat'])
    lon1 = radians(c1['lon'])
    lat2 = radians(c2['lat'])
    lon2 = radians(c2['lon'])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    return distance


# In[29]:


def distProb(dist: float):
    return 1 - math.tanh(dist / 10)


# In[65]:


from model import *

def recommendFiltersd3LevelIds(filters):
    
    filtersActivities = {}

    if ('d0LevelId' in filters)and(filters['d0LevelId'] is not None):
        filtersActivities['d0LevelName'] = {'$in': filters['d0LevelId']}

    if ('d1LevelId' in filters)and(filters['d1LevelId'] is not None):
        filters['d1LevelId'] = [int(id) for id in filters['d1LevelId']]
        filtersActivities['d1LevelId'] = {'$in': filters['d1LevelId']}

    if ('d2LevelId' in filters)and(filters['d2LevelId'] is not None):
        filters['d2LevelId'] = [int(id) for id in filters['d2LevelId']]
        filtersActivities['d2LevelId'] = {'$in': filters['d2LevelId']}
        
    if ('d3LevelId' in filters)and(filters['d3LevelId'] is not None):
        filters['d3LevelId'] = [int(id) for id in filters['d3LevelId']]
        filtersActivities['d3LevelId'] = {'$in': filters['d3LevelId']}

    if ('online' in filters)and(filters['online'] is not None):
        filtersActivities['online'] = filters['online']

    if ('certificate' in filters)and(filters['certificate'] is not None):
        filtersActivities['certificate'] = filters['certificate']
                
    if len(filtersActivities) > 0:
        d3LevelIds = set([i['d3LevelId'] for i in db.activities.find(filtersActivities, {'d3LevelId': 1})])
    else:
        d3LevelIds = None

    d3LevelRecsRelevance = None

    filtersSchedule = {}

    # Тут просто завод по производству костылей, но так удобнее для метода поиска
    if ('days' in filters)and(filters['days'] is not None):
        filters['days'] = [int(id) for id in filters['days']]
        filtersSchedule['schedule.day'] = {'$in': [DAYS_ID_DICT[i] for i in filters['days']]}

    if ('district' in filters)and(filters['district'] is not None):
        filters['district'] = [int(id) for id in filters['district']]
        filtersSchedule['district'] = {'$in': [DISTRICT_DICT[i] for i in filters['district']]}

    if ('area' in filters)and(filters['area'] is not None):
        filters['area'] = [int(id) for id in filters['area']]
        filtersSchedule['area'] = {'$in': [AREA_DICT[i] for i in filters['area']]}

    if ('active' in filters)and(filters['active'] is not None):
        filtersSchedule['active'] = filters['active']
        
    if d3LevelIds is not None:
        filtersSchedule['d3LevelId'] = {'$in': list(d3LevelIds)}
                
    if len(filtersSchedule) == 0:
        return None
    
    res = list(db.schedule.find(filtersSchedule, {'_id': False}))
        
    return list(set([r['d3LevelId'] for r in res if 'd3LevelId' in r]))

def recommendFromVector(probVec, n: int = 5, randomChoice = False):

    if randomChoice:
        res = zip(
            np.random.choice(list(range(probVec.shape[0])), n, p = probVec, replace = False),
            [1 for _ in range(n)]
        )
    else:
        res = zip(np.argsort(probVec)[::-1][:n], np.sort(probVec)[::-1][:n])
        
    return list(res)

def recommendFilters(filters, userId, recBestLimit = 5, recRareLimit = 5):
    
    modelMain = ModelMain(modelPath = './model/modelMain')
    modelAge = ModelMeta(db = db, modelPath = './model/modelAge')

    d3Id2Idx = modelMain.d3Id2Idx
    d3Id2IdxReverse = {v : k for k, v in d3Id2Idx.items()}

    
    probVecBest, probVecRare = modelMain.recomendForUser(userId)
    metaVec = modelAge(userId)
    
    d3LevelIds = recommendFiltersd3LevelIds(filters)
    
    if d3LevelIds is not None:
                            
        mask = np.zeros(probVecBest.shape)
        mask[[d3Id2Idx[id] for id in d3LevelIds if id in d3Id2Idx]] = 1
            
        probVecBest *= mask
        probVecRare *= mask
        metaVec *= mask
            
        probVecBest = probVecBest / np.sum(probVecBest)
        probVecRare = probVecRare / np.sum(probVecRare)
        metaVec = metaVec / np.sum(metaVec)

    recBest = recommendFromVector(multiplyProbVecs(probVecBest, metaVec), n = recBestLimit)
    recRare = recommendFromVector(multiplyProbVecs(probVecRare, metaVec), n = recRareLimit)
    
    recBest = [[d3Id2IdxReverse[r], p] for r, p in recBest]
    recRare = [[d3Id2IdxReverse[r], p] for r, p in recRare]
    
    return recBest + recRare

def recommendFiltersGroups(filters, userId, limit = 10, offset = 0):
    recs = recommendFilters(filters, userId)
        
    res = []
    
# DEBUG|
    
#     print('----Посещенные курсы----')
#     for i in set([a['d3LevelId'] for a in getAttends(userId)]):
#         print(i, db.activities.find_one({'d3LevelId': i})['d3LevelName'])

#     print('----Рекомендованные курсы----')
#     for i in recs:
#         print(i, db.activities.find_one({'d3LevelId': int(i[0])})['d3LevelName'])
    
    groupsPerRec = [
        list(db.schedule.find({
            'd3LevelId': int(i[0]),
            'addressCoords': {'$exists': True},
             'active': filters['active'] if ('active' in filters)and(filters['active'] is not None) else False
        }, {
            'groupId': 1, 'addressCoords': 1, 'd3LevelId': 1, '_id': False
        })) for i in recs
    ]
    
    userCoords = getUserCoords(userId)
        
    for i in range(len(groupsPerRec)):
        
        online = db.activities.find_one({'d3LevelId': int(recs[i][0])})['online']
        
        for j in range(len(groupsPerRec[i])):
                        
            if online:
                # Это сделано для того, чтобы больше рекомендовать оффлайн курсы, можно регулировать 
                # настойчивость данных рекомендаций
                pDist = 0.5
            else:
                pDist = distProb(calcDistance(userCoords, groupsPerRec[i][j]['addressCoords']))
                            
            groupsPerRec[i][j] = {
                'groupId': groupsPerRec[i][j]['groupId'],
                'p': pDist * recs[i][1]
            }
            
        groupsPerRec[i] = sorted(groupsPerRec[i], key = lambda x: -x['p'])
        
        probSet = set()
        
        groupPerRecDupsFiltered = []
        
        for g in groupsPerRec[i]:
            if g['p'] not in probSet:
                probSet.add(g['p'])
                groupPerRecDupsFiltered.append(g)
            else:
                pass
            
        groupsPerRec[i] = groupPerRecDupsFiltered[:3]
        
    groups = []
    
    for gg in groupsPerRec:
        for g in gg:
            groups.append(g)
                        
    return [
        db.schedule.find_one({'groupId': g['groupId']}, {'_id': False})
        for g in groups
    ]


# In[66]:


def encoder(d):
    
    if isinstance(d, dict):
        return {k: encoder(v) for k, v in d.items()}
        
    if isinstance(d, list):
        return [encoder(dd) for dd in d]
        
    if isinstance(d, float):
        if math.isnan(d):
            return None
        return d
    
    if isinstance(d, datetime):
        return d.strftime(DATEFORMAT)
    
    if isinstance(d, time):
        return d.strftime('%H:%M')
            
    return d


# In[67]:


class NoParameterFoundException(Exception):
    pass

def req(content, param):
    if param not in content:
        raise NoParameterFoundException(param)
    return content[param]

def options_wrapper(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            return get_response('ok')
        else:
            return f(*args, **kwargs)
    return decorated_function

def token_wrapper(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if ('token' not in request.json):
            logging.debug('No token in request')
            
            return get_response({
                    'err': 'Malformed request'
                })
        else:
            if db.users.find_one({
                'token': request.json['token']
            }) is None:
                
                logging.debug('Invalid token: %s'%request.json['token'])
                
                return get_response({
                    'err': 'Malformed request'
                })
            else:
                logging.debug('Valid token: %s'%request.json['token'])
                
                return f(*args, **kwargs)
    
    return decorated_function
        
def get_response(response):
    
    if isinstance(response, dict):
        #response = remove_nans_from_json(response)
        response = make_response(
            jsonify(encoder(response)) if isinstance(response, dict) else response
        )

        response.headers['Content-Type'] = 'application/json'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = '*'
        
        return response
        
    elif (isinstance(response, str)):
        response = make_response(
            response
        )
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = '*'
        
        return response
    
    return response

# Старая версия, просто рекомендуем что попало

def getRecommendationsByD0():

    d0LevelGroups = ['Для ума', 'Для души', 'Для тела']

    res = [
        {
            'typeGroup': g,
            'activities': [
                {'title': i['d3LevelName'], 'd3LevelId': i['d3LevelId']} 
                for i in list(db.activities.find({'d0LevelName': g}).limit(5))
            ]

        } for g in d0LevelGroups
    ]
    
    return res

# Новая версия с рекомендательной системой

def getRecommendationsByD0V2(userId: int):

    d0LevelGroups = ['Для ума', 'Для души', 'Для тела']
    
    res = [
        {
            'typeGroup': g,
            'activities': [
                {'d3LevelId': i, 'title': db.activities.find_one({'d3LevelId': i})['d3LevelName']}
                for i in 
                list(
                    set([i['d3LevelId'] for i in recommendFiltersGroups({'d0LevelId': [g]}, userId)])
                )[:5]]
        } for g in d0LevelGroups
    ]
    
    return res


# In[68]:


import sys

sys.path.append('/home/immichail/anaconda3/lib/python3.8/site-packages')

from fuzzysearch import find_near_matches


def fuzzySearch(q, d3LevelRecs: list):
    d3LevelNamesSearch = []

    for n in d3LevelRecs:
        try:
            minDist = sum([
                sorted(
                    find_near_matches(qq.lower(), n['d3LevelName'].lower(), max_l_dist = len(qq) - 1), 
                    key = lambda x: x.dist
                )[0].dist
                for qq in q.split()
            ])

            d3LevelNamesSearch.append({
                'd3LevelId': n['d3LevelId'], 
                'd3LevelName': n['d3LevelName'], 
                'minDist': minDist
            })
        except:
            pass
                
    return d3LevelNamesSearch

DAYS_ID_DICT = [{'id': 0, 'label': 'Пн'},
 {'id': 1, 'label': 'Вт'},
 {'id': 2, 'label': 'Ср'},
 {'id': 3, 'label': 'Чт'},
 {'id': 4, 'label': 'Пт'},
 {'id': 5, 'label': 'Сб'},
 {'id': 6, 'label': 'Вс'}]
DAYS_ID_DICT = {
    item['id']: item['label'] for item in DAYS_ID_DICT
}

DISTRICT_DICT = db.vars.find_one({'id': 'districtDict'})['value']
DISTRICT_DICT = {
    item['id']: item['name'] for item in DISTRICT_DICT
}

AREA_DICT = db.vars.find_one({'id': 'areaDict'})['value']
AREA_DICT = {
    item['id']: item['name'] for item in AREA_DICT
}


def searchFilters(filters, limit = 10, offset = 0):
    
    print(filters)
    
    filtersActivities = {}

    if ('d0LevelId' in filters)and(filters['d0LevelId'] is not None):
        filtersActivities['d0LevelName'] = {'$in': filters['d0LevelId']}

    if ('d1LevelId' in filters)and(filters['d1LevelId'] is not None):
        filters['d1LevelId'] = [int(id) for id in filters['d1LevelId']]
        filtersActivities['d1LevelId'] = {'$in': filters['d1LevelId']}

    if ('d2LevelId' in filters)and(filters['d2LevelId'] is not None):
        filters['d2LevelId'] = [int(id) for id in filters['d2LevelId']]
        filtersActivities['d2LevelId'] = {'$in': filters['d2LevelId']}
        
    if ('d3LevelId' in filters)and(filters['d3LevelId'] is not None):
        filters['d3LevelId'] = [int(id) for id in filters['d3LevelId']]
        filtersActivities['d3LevelId'] = {'$in': filters['d3LevelId']}

    if ('online' in filters)and(filters['online'] is not None):
        filtersActivities['online'] = filters['online']

    if ('certificate' in filters)and(filters['certificate'] is not None):
        filtersActivities['certificate'] = filters['certificate']
        
    if len(filtersActivities) > 0:
        d3LevelIds = set([i['d3LevelId'] for i in db.activities.find(filtersActivities, {'d3LevelId': 1})])
    else:
        d3LevelIds = None

    if ('q' in filters)and(filters['q'] is not None):
        if d3LevelIds is None:
            d3LevelRecs = list(db.activities.find({}, {'d3LevelId': 1, 'd3LevelName': 1}))
        else:
            d3LevelRecs = list(db.activities.find({'d3LevelId': {'$in': list(d3LevelIds)}}, {'d3LevelId': 1, 'd3LevelName': 1}))

        d3LevelRecsRelevance = fuzzySearch(filters['q'], d3LevelRecs)

        d3LevelRecsRelevance = {i['d3LevelId']: i['minDist'] for i in d3LevelRecsRelevance}
    else:
        d3LevelRecsRelevance = None

    filtersSchedule = {}

    # Тут просто завод по производству костылей, но так удобнее для метода поиска
    if ('days' in filters)and(filters['days'] is not None):
        filters['days'] = [int(id) for id in filters['days']]
        filtersSchedule['schedule.day'] = {'$in': [DAYS_ID_DICT[i] for i in filters['days']]}

    if ('district' in filters)and(filters['district'] is not None):
        filters['district'] = [int(id) for id in filters['district']]
        filtersSchedule['district'] = {'$in': [DISTRICT_DICT[i] for i in filters['district']]}

    if ('area' in filters)and(filters['area'] is not None):
        filters['area'] = [int(id) for id in filters['area']]
        filtersSchedule['area'] = {'$in': [AREA_DICT[i] for i in filters['area']]}
        
    if ('active' in filters)and(filters['active'] is not None):
        filtersSchedule['active'] = filters['active']

    if d3LevelIds is not None:
        filtersSchedule['d3LevelId'] = {'$in': list(d3LevelIds)}
                
    res = list(db.schedule.find(filtersSchedule, {'_id': False}))
        
    if d3LevelRecsRelevance is not None:
        
#         d3LevelIdsRelevanceDict = {r['d3LevelId']: r['minDist'] for r in d3LevelRecsRelevance}
        
        res = [r for r in res if ('d3LevelId' in r)and(r['d3LevelId'] in d3LevelRecsRelevance)]
        res = sorted(res, key = lambda x: d3LevelRecsRelevance[x['d3LevelId']])

    # res = random.sample(res, limit) #[offset:offset + limit]
    
    res = sorted(res, key = lambda x: x['groupId'])[offset:offset + limit]

    return res


# In[69]:


# def fuzzySearch(q, d3LevelNames: list):
#     d3LevelNamesSearch = []

#     for n in d3LevelNames:
#         try:
#             minDist = sum([
#                 sorted(find_near_matches(qq.lower(), n.lower(), max_l_dist = len(qq) - 1), key = lambda x: x.dist)[0].dist
#                 for qq in q.split()
#             ])

#             d3LevelNamesSearch.append({
#                 'name': n, 
#                 'minDist': minDist
#             })
#         except:
#             pass
        
#     return d3LevelNamesSearch

# def searchFilters(filters, limit = 10, offset = 0):

#     filtersActivities = {}

#     if ('d0LevelName' in filters)and(filters['d0LevelName'] is not None):
#         filtersActivities['d0LevelName'] = {'$in': filters['d0LevelName']}

#     if ('d1LevelName' in filters)and(filters['d1LevelName'] is not None):
#         filtersActivities['d1LevelName'] = {'$in': filters['d1LevelName']}

#     if ('d2LevelName' in filters)and(filters['d2LevelName'] is not None):
#         filtersActivities['d2LevelName'] = {'$in': filters['d2LevelName']}

#     if ('online' in filters)and(filters['online'] is not None):
#         filtersActivities['online'] = filters['online']

#     if ('certificate' in filters)and(filters['certificate'] is not None):
#         filtersActivities['certificate'] = filters['certificate']

#     if len(filtersActivities) > 0:
#         d3LevelNames = set([i['d3LevelName'] for i in db.activities.find(filtersActivities, {'d3LevelName': 1})])
#     else:
#         d3LevelNames = None

#     if ('q' in filters)and(filters['q'] is not None):
#         if d3LevelNames is None:
#             d3LevelNames = set([i['d3LevelName'] for i in db.activities.find({}, {'d3LevelName': 1})])

#         d3LevelNamesRelevance = fuzzySearch(filters['q'], d3LevelNames)

#         d3LevelNamesRelevance = {i['name']: i['minDist'] for i in d3LevelNamesRelevance}
#     else:
#         d3LevelNamesRelevance = None

#     filtersSchedule = {}

#     if ('days' in filters)and(filters['days'] is not None):
#         filtersSchedule['schedule.day'] = {'$in': filters['days']}

#     if ('district' in filters)and(filters['district'] is not None):
#         filtersSchedule['district'] = {'$in': filters['district']}

#     if ('area' in filters)and(filters['area'] is not None):
#         filtersSchedule['area'] = {'$in': filters['area']}

#     if d3LevelNames is not None:
#         filtersSchedule['d3LevelName'] = {'$in': list(d3LevelNames)}
        
#     res = list(db.schedule.find(filtersSchedule, {'_id': False}))

#     if d3LevelNamesRelevance is not None:
#         res = [r for r in res if r['d3LevelName'] in d3LevelNamesRelevance]
#         res = sorted(res, key = lambda x: d3LevelNamesRelevance[x['d3LevelName']])

#     res = res[offset:offset + limit]

#     return res

def getCategories(param: str):
    if param in ['d0LevelName', 'd1LevelName', 'd2LevelName']:
        return list(set(db.activities.distinct(param)))
    if param in ['area', 'district']:
        return list(set(sorted([i.strip() for i in db.schedule.distinct(param) if (isinstance(i, str))and(i.strip() != '')])))
    if param in ['online', 'certificate']:
        return ['Да', "Нет"]
    if param in ['days']:
        return ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
    
def getCategories(param: str):
    
    if param in ['d0LevelId']:
        return [
            {'value': i, 'label': i}
            for i in list(set(db.activities.distinct('d0LevelName')))
        ]
    
    if param in ['d1LevelId', 'd2LevelId']:
        return [
            {'value': i, 'label': db.activities.find_one({param: i})[param.replace('Id', 'Name')]}
            for i in list(set(db.activities.distinct(param)))
        ]   
                   
    if param in ['area', 'district']:
        
        d = db.vars.find_one({'id': '%sDict'%param})['value']
        d = {item['name']: item['id'] for item in d}
        
        return [
            {'value': d[i], 'label': i}
            for i in list(
                set(sorted([i.strip() for i in db.schedule.distinct(param) if (isinstance(i, str))and(i.strip() != '')]))
            )
        ]
        
    if param in ['online', 'certificate']:
        return [{'value': 0, 'label': 'Да'}, {'value': 1, 'value': "Нет"}]
    if param in ['days']:
        return [
            {'value': i, 'label': k}
            for i, k in enumerate(['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'])
        ]
    
def groupAvailableDates(
    groupId,
    dateStart,
    dateEnd
):

    rec = db.schedule.find_one({'groupId': groupId})
    dateStartDt = datetime.strptime(dateStart, DATEFORMAT)
    dateEndDt = datetime.strptime(dateEnd, DATEFORMAT)
    
    availableDates = []

    for item in rec['schedule']:
        
        dates = []

        startShift = DAYS_DICT[item['day']] - (dateStartDt).weekday()
        if startShift < 0:
            startShift += 7

        d = dateStartDt + timedelta(days = startShift)

        while (d < dateEndDt):
            dates.append(d.strftime(DATEFORMAT))
            d += timedelta(days = 7)

        availableDates.extend([
            {'day': item['day'],'date': d, 'timeStart': item['timeStart'], 'timeEnd': item['timeEnd']} for d in dates
        ])
    
    return availableDates


# In[70]:


def getWorkId(groupId, date, timeStart):
    workRec = db.works.find_one({'groupId': groupId, 'date': date, 'timeStart': timeStart})

    if workRec is not None:
        return workRec['workId']
    else:

        if (db.works.find_one({}) is not None):
            workMax = max(
                list(db.attends.find({}).sort([("workId", pymongo.DESCENDING)]).limit(1))[0]['workId'],
                list(db.works.find({}).sort([("workId", pymongo.DESCENDING)]).limit(1))[0]['workId']
            )
        else:
            workMax = list(db.attends.find({}).sort([("workId", pymongo.DESCENDING)]).limit(1))[0]['workId']

        workMax = workMax + 1

        db.works.insert_one({
            'workId': workMax,
            'groupId': groupId, 
            'date': date, 
            'timeStart': timeStart
        })

        return workMax
    
def signAttend(userId, groupId, date, timeStart, timeEnd):

    if len(timeStart.split(':')) == 3:
        timeStart = ':'.join(timeStart.split(':')[:-1])

    if len(timeEnd.split(':')) == 3:
        timeEnd = ':'.join(timeEnd.split(':')[:-1])

    workId = getWorkId(groupId, date, timeStart)
    
    scheduleRec = db.schedule.find_one({'groupId': groupId})
    activityRec = db.activities.find_one({'d3LevelId': scheduleRec['d3LevelId']})
    
    attend = {
        'workId': workId,
        'groupId': groupId,
        'userId': userId,
        'd2LevelName': scheduleRec['d2LevelName'],
        'd3LevelName': scheduleRec['d3LevelName'],
        'online': activityRec['online'],
        'date': datetime.strptime(date, DATEFORMAT),
        'timeStarted': timeStart,
        'timeEnded': timeEnd,
        'd0LevelName': activityRec['d0LevelName'],
        'd1LevelId': activityRec['d1LevelId'],
        'd2LevelId': activityRec['d2LevelId'],
        'd3LevelId': activityRec['d3LevelId'],
        'status': 'planned'
    }
    
    db.attends.insert_one(attend)
    
    return attend


# In[ ]:


app = Flask(__name__)
app.config['DEBUG'] = True
        
@app.route('/status', methods = ['GET', 'POST', 'OPTIONS'])
@options_wrapper
def getStatusMethod():
    content = request.json
        
    return get_response({
        'res': 'ok'
    })

@app.route('/user/set', methods = ['GET', 'POST', 'OPTIONS'])
@options_wrapper
def userSetMethod():
    content = request.json
    
    if 'uuid' not in content:
        content['uuid'] = uuid()
        db.users.insert_one(content)
    else:
        db.users.update_one({'uuid': content['uuid']}, {'$set': content})
        
    return get_response({
        'res': content['uuid']
    })

@app.route('/user/get', methods = ['GET', 'POST', 'OPTIONS'])
@options_wrapper
def userGetMethod():
    content = request.json
    
    userRecord = db.users.find_one({'uuid': content['uuid']})
    
    if userRecord is None:
        return get_response({
            'err': 'User does not exists'
        })
    else:
        del userRecord['_id']
        return get_response({
            'res': userRecord
        })
    
@app.route('/userV2/get', methods = ['GET', 'POST', 'OPTIONS'])
@options_wrapper
def userV2GetMethod():
    content = request.json
    
    userId = request.args.get('userId', type = int)
    
    if userId is not None:
        userId = int(userId)
        
    if userId is not None:
        userRecord = db.usersV2.find_one({
            'userId': userId
        })

        if userRecord is None:
            return get_response({
                'err': 'User does not exists'
            })
        else:
            del userRecord['_id']
            return get_response({
                'res': userRecord
            })

    
    name = req(content, 'name')
    surName = req(content, 'surName')
    thirdName = req(content, 'thirdName')
    dateBirth = req(content, 'dateBirth')
    dateBirth = parseDate(dateBirth)
    
    userRecord = db.usersV2.find_one({
        'name': name,
        'surName': surName,
        'thirdName': thirdName,
        'dateBirth': dateBirth
    })
    
    if userRecord is None:
        return get_response({
            'err': 'User does not exists'
        })
    else:
        del userRecord['_id']
        return get_response({
            'res': userRecord
        })

# @app.route('/userV2/set', methods = ['GET', 'POST', 'OPTIONS'])
# @options_wrapper
# def userV2SetMethod():
#     content = request.json
    
#     name = req(content, 'name')
#     surName = req(content, 'surName')
#     thirdName = req(content, 'thirdName')
#     dateBirth = req(content, 'dateBirth')
#     dateBirth = parseDate(dateBirth)
#     gender = req(content, 'gender')
#     address = req(content, 'address')
#     dateCreated = datetime.now()
    
#     userId = list(db.usersV2.find().sort('userId', -1).limit(1))[0]['userId']
        
#     db.usersV2.insert_one({
#         'userId': userId,
#         'dateCreated': dateCreated,
#         'dateBirth': dateBirth,
#         'address': address,
#         'name': name,
#         'surName': surName,
#         'thirdName': thirdName,
#         'gender': gender
#     })
    
#     return get_response({
#         'res': 'ok'
#     })

@app.route('/group/<groupId>', methods = ['GET', 'POST', 'OPTIONS'])
@options_wrapper
def groupGetMethod(groupId: int):
    content = request.json
    
    res = db.schedule.find_one({'groupId': int(groupId)}, {'_id': False})
    
    return get_response({
        'res': res
    })

@app.route('/work/<workId>/groupList', methods = ['GET', 'POST', 'OPTIONS'])
@options_wrapper
def workGoupListMethod(workId: int):
    content = request.json
    
    res = list(db.schedule.find({'d3LevelName': db.activities.find_one({'d3LevelId': workId})['d3LevelName']}))
    
    return get_response({
        'res': res
    })

# q = <любая строка>,
# dxLevelName = <act1>;<act2>;...<act3>
# online = Да|Нет
# certificate = Да|Нет
# days = <day1>;<day2>;...<day3>
# districts = <districts1>;<districts2>;...<districts3>
# area = <area1>;<area2>;...<area3>

def ifSplit(s):
    if s is None:
        return s
    return s.split(';')

@app.route('/search', methods = ['GET', 'POST', 'OPTIONS'])
@options_wrapper
def searchMethod():
    content = request.json
    
    userId = request.args.get('userId', type = int)
    if userId is not None:
        userId = int(userId)
    
    limit = request.args.get('limit', default = 10, type = int)
    offset = request.args.get('offset', default = 0, type = int)
    
    q = request.args.get('q', type = str)
    
    d0LevelName = ifSplit(request.args.get('d0LevelId', type = str))
    d1LevelName = ifSplit(request.args.get('d1LevelId', type = str))
    d2LevelName = ifSplit(request.args.get('d2LevelId', type = str))
    d3LevelName = ifSplit(request.args.get('d3LevelId', type = str))
    
    online = request.args.get('online', type = str)
    if online is not None:
        online = online == '0'
    certificate = request.args.get('certificate', type = str)
    if certificate is not None:
        certificate = certificate == '0'
    online = request.args.get('online', type = str)
    if online is not None:
        online = online == '0'
        
    active = request.args.get('active', type = str)
    if active is not None:
        active = active == '1'
    
    days = ifSplit(request.args.get('days', type = str))
    district = ifSplit(request.args.get('district', type = str))
    area = ifSplit(request.args.get('area', type = str))
    
    filters = {
        'q': q,
        'd0LevelId': d0LevelName,
        'd1LevelId': d1LevelName,
        'd2LevelId': d2LevelName,
        'd3LevelId': d3LevelName,
        'online': online,
        'certificate': certificate,
        'days': days,
        'district': district,
        'area': area,
        'active': active
    }
    
    if (('q' not in filters)or(filters['q'] is None))and(userId is not None):
        res = recommendFiltersGroups(filters, userId)
    else:
        res = searchFilters(filters, limit, offset)
    
    
    return get_response({
        'searchActivities': res
    })


@app.route('/startPageRecommendations', methods = ['GET', 'POST', 'OPTIONS'])
@options_wrapper
def startPageRecommendationsMethod():
    content = request.json
    
       
    userId = request.args.get('userId', default = None, type = int)
    
    if userId is None:
        return get_response({
            'recomendedActivities': getRecommendationsByD0()
        })
    else:
        return get_response({
            'recomendedActivities': getRecommendationsByD0V2(int(userId))
        })

@app.route('/filter/values/<filterName>', methods = ['GET', 'POST', 'OPTIONS'])
@options_wrapper
def getFilterValuesMethod(filterName: str):
    content = request.json
      
    return get_response({
        filterName: [i for i in getCategories(filterName)]
    })

@app.route('/var/<varName>', methods = ['GET', 'POST', 'OPTIONS'])
@options_wrapper
def getSurveyMethod(varName:str):
    content = request.json
    
    rec = db.vars.find_one({'id': varName})
    
    if rec is not None:
        return get_response(rec['value'])
    
    return get_response({
        'err': "No such var"
    })

@app.route('/group/<groupId>/availableDates', methods = ['GET', 'POST', 'OPTIONS'])
@options_wrapper
def getGroupAvailableDates(groupId: int):
    content = request.json
    
    dateStart = request.args.get(
        'dateStart', 
        default = (datetime.now() + timedelta(days = 1)).strftime(DATEFORMAT), 
        type = str
    )
    dateEnd = request.args.get(
        'dateEnd', 
        default = (datetime.now() + timedelta(days = 15)).strftime(DATEFORMAT), 
        type = str
    )
    
    groupId = int(groupId)

    return get_response({
        'res': groupAvailableDates(groupId, dateStart, dateEnd)
    })


@app.route('/user/<userId>/sign/<groupId>', methods = ['GET', 'POST', 'OPTIONS'])
@options_wrapper
def signUserForGroup(userId:int, groupId: int):
    content = request.json
    
    userId = int(userId)
    groupId = int(groupId)
    date = content['date']
    timeStart = content['timeStart']
    timeEnd = content['timeEnd']
    
    attend = signAttend(userId, groupId, date, timeStart, timeEnd)
    
    del attend['_id']
        
    return get_response({
        'res': attend
    })

@app.route('/user/<userId>/attends', methods = ['GET', 'POST', 'OPTIONS'])
@options_wrapper
def getAttendsForUser(userId: int):
    content = request.json
    
    userId = int(userId)
    
    attends = list(db.attends.find({'userId': userId}, {'_id': False}).sort([('date', pymongo.DESCENDING)]))
    
    for a in attends:
        if 'status' not in a:
            a['status'] = 'visited'
    
    return get_response({
        'res': attends
    })

@app.route('/userV2/set', methods = ['GET', 'POST', 'OPTIONS'])
@options_wrapper
def userV2SetMethod():
    content = request.json
    
    name, surName, thirdName = req(content, 'fio').split(' ')
    dateBirth = parseDate(req(content, 'date'))
    gender = req(content, 'sex')
    
    rec = db.usersV2.find_one({
        'name': name,
        'surName': surName,
        'thirdName': thirdName,
        'dateBirth': dateBirth
    })
        
    if rec is None:
        userId = list(db.usersV2.find().sort('userId', -1).limit(1))[0]['userId'] + 1
        dateCreated = datetime.now()
        
        db.usersV2.insert_one({
            'userId': userId,
            'dateCreated': dateCreated,
            'dateBirth': dateBirth,
            'address': '',
            'name': name,
            'surName': surName,
            'thirdName': thirdName,
            'gender': gender
        })

        return get_response({
            'res': userId
        })
    else:
        return get_response({
            'res': rec['userId']
        })
    
#     name = req(content, 'name')
#     surName = req(content, 'surName')
#     thirdName = req(content, 'thirdName')
#     dateBirth = req(content, 'dateBirth')
#     dateBirth = parseDate(dateBirth)
#     gender = req(content, 'gender')
#     address = req(content, 'address')
#     dateCreated = datetime.now()
    
#     userId = list(db.usersV2.find().sort('userId', -1).limit(1))[0]['userId']
        
#     db.usersV2.insert_one({
#         'userId': userId,
#         'dateCreated': dateCreated,
#         'dateBirth': dateBirth,
#         'address': address,
#         'name': name,
#         'surName': surName,
#         'thirdName': thirdName,
#         'gender': gender
#     })
    
    return get_response({
        'res': 'ok'
    })


run_simple('localhost', 11059, app)


# In[ ]:





































# ### 
