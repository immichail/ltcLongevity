import pickle as pkl
import numpy as np
from datetime import datetime
import pymongo

db = pymongo.MongoClient()['ltcLongevity']

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

DATE_ATTENUATION_PARAM = 0.9

def ageClassFunction(user = None, userId = None):
    if user is None:
        user = db.usersV2.find_one({'userId': userId})
    
    age = (datetime.now() - user['dateBirth']).days / 365.25
    if age < 60:
        return '<60'
    if age > 90:
        return '>90'
    return '%d-%d'%(age // 10  * 10, (age // 10 + 1)  * 10)

class ModelMeta:
    
    def __init__(self, db = None, d3Id2Idx = None, modelPath = None):
        
        if (db is None):
            raise Exception('Model need DB connection to work')
            
        self.db = db
        self.d3Id2Idx = d3Id2Idx
        
        if modelPath is not None:
            self.d3Id2Idx = pkl.load(open(f'{modelPath}/d3Id2Idx.pkl', 'rb'))
            self.classFunction = pkl.load(open(f'{modelPath}/classFunction.pkl', 'rb'))
            self.dictClasses = pkl.load(open(f'{modelPath}/dictClasses.pkl', 'rb'))
        
        return None
        
    def fit(self, dfm, classFunction):
        
        self.classFunction = classFunction
        
        self.classes = set()

        self.userClasses = {}
        users = list(self.db.usersV2.find({}))
        
        for user in tqdm(users):
            cls = self.classFunction(user = user)
            self.classes.add(cls)
            self.userClasses[user['userId']] = cls
            
        dfClasses = pd.DataFrame([{'userId': k, 'cls': v} for k, v in self.userClasses.items()])
        dfClasses = df.merge(dfClasses, on = 'userId', how = 'inner')
        dictClasses = dfClasses.groupby('cls').agg({'d3LevelId': list}).to_dict(orient = 'index')
        
        for k in dictClasses:
            vec = np.zeros(len(self.d3Id2Idx))

            for i in dictClasses[k]['d3LevelId']:
                vec[self.d3Id2Idx[i]] += 1

            dictClasses[k] = vec / np.sum(vec)
            
        self.dictClasses = dictClasses
        
        return None
    
    def save(self, modelPath):
        try:
            os.mkdir(modelPath)
        except:
            logging.exception('Unable to create folder for model')
            
        pkl.dump(self.classFunction, open(f'{modelPath}/classFunction.pkl', 'wb'))
        pkl.dump(self.dictClasses, open(f'{modelPath}/dictClasses.pkl', 'wb'))
    
    def __call__(self, userId: str):
        cls = self.classFunction(userId = userId)
        return self.dictClasses[cls]

def multiplyProbVecs(v1, v2):
    vm = v2 / (1 - v2)
    v1 *= vm
    v1 = v1 / v1.sum()
    return v1

def getAttends(userId: str):
    
#     recs = df[df['userId'] == userId].to_dict(orient = 'records')
    recs = db.attends.find({'userId': userId})
    
    return [
        {'d3LevelId': r['d3LevelId'], 'date': r['date']} for r in recs
    ]

class ModelMain():
    
    def __init__(self, d3LevelCounts = None, probTotal = None, d3Id2Idx = None, modelPath = None):
        self.d3LevelCounts = d3LevelCounts
        self.d3Id2Idx = d3Id2Idx
        self.probTotal = probTotal
        
        if modelPath is not None:
            self.d3LevelCounts = pkl.load(open(f'{modelPath}/d3LevelCounts.pkl', 'rb'))
            self.probTotal = pkl.load(open(f'{modelPath}/probTotal.pkl', 'rb'))
            self.d3Id2Idx = pkl.load(open(f'{modelPath}/d3Id2Idx.pkl', 'rb'))
        
    def recomendForUser(
        self,
        userId: int, 
        best: int = 5, 
        rare: int = 5,
        randomChoice = False,
        dateAttenuation = True
    ):

        attends = getAttends(userId)
        
        probVecBest = self.recommendVectorBest(attends, dateAttenuation = dateAttenuation)
        probVecRare = self.recommendVectorRare(attends, dateAttenuation = dateAttenuation)

        for a in attends:
            probVecBest[self.d3Id2Idx[a['d3LevelId']]] = 0
            probVecRare[self.d3Id2Idx[a['d3LevelId']]] = 0


        return probVecBest, probVecRare

    def recommendFromVector(probVec, n: int = 5, randomChoice = False):

        if randomChoice:
            res = np.random.choice(list(range(probVec.shape[0])), n, p = probVec, replace = False)
        else:
            res = np.argsort(probVec)[::-1][:n]

    #         def getSetProb(arr):

    #         p = 1.0

    #         for i in range(len(arr)):
    #             for j in range(i + 1, len(arr)):
    #                 p *= d3LevelCounts[arr[i]][arr[j]]

    #         return p

    #     if randomChoice:
    #         reses = [
    #             np.random.choice(list(range(probVec.shape[0])), n, p = probVec, replace = False)
    #             for i in range(10)
    #         ]

    #         reses = [(r, getSetProb(r)) for r in reses]

    #         res = sorted(reses, key = lambda x: x[1])[0][0]
    #     else:
    #         res = np.argsort(probDiff)[::-1][:n]

        return list(res)

    def recommendVectorBest(self, attends: list, dateAttenuation = False):

        attends = sorted(attends, key = lambda x: x['date'])[::-1]

        probVec = np.array([self.d3LevelCounts[self.d3Id2Idx[a['d3LevelId']]] for a in attends])

        if dateAttenuation:
            dateMult = []

            m = 1
            for i in range(0, len(attends)):
                dateMult.append(m)
                m *= DATE_ATTENUATION_PARAM
        else:
            dateMult = np.ones(len(attends))

        probVec = probVec.transpose().dot(dateMult)
        probVec /= np.sum(probVec)

        return probVec

    def recommendVectorRare(self, attends: list, dateAttenuation = False):

        attends = sorted(attends, key = lambda x: x['date'])[::-1]

        probVec = np.array([self.d3LevelCounts[self.d3Id2Idx[a['d3LevelId']]] for a in attends])

        if dateAttenuation:
            dateMult = []

            m = 1
            for i in range(0, len(attends)):
                dateMult.append(m)
                m *= DATE_ATTENUATION_PARAM
        else:
            dateMult = np.ones(len(attends))

        probVec = probVec.transpose().dot(dateMult)
        probVec /= np.sum(probVec)

        probDiff = (probVec - self.probTotal)
        probDiff[np.where(probDiff > 0)] = 0
        probDiff = -probDiff
        probDiff /= probDiff.sum()

        return probDiff

    def recommendForUserNew(self):
        return None
    
    def save(self, modelPath):
        try:
            os.mkdir(modelPath)
        except:
            logging.exception('Unable to create folder for model')
            
        pkl.dump(self.d3LevelCounts, open(f'{modelPath}/d3LevelCounts.pkl', 'wb'))
        pkl.dump(self.probTotal, open(f'{modelPath}/probTotal.pkl', 'wb'))
        pkl.dump(self.d3Id2Idx, open(f'{modelPath}/d3Id2Idx.pkl', 'wb'))
        
def recommendFromVector(probVec, n: int = 5, randomChoice = False):

    if randomChoice:
        res = np.random.choice(list(range(probVec.shape[0])), n, p = probVec, replace = False)
    else:
        res = np.argsort(probVec)[::-1][:n]
        
    return list(res)

