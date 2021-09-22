import torch
import datetime
import os
import errno

def save(folderName, project, variablesToSave):
    folder = makeFolderOfTheDay(folderName)
    subFolder = makeSubfolder(folder, project)

    allVariables = list(variablesToSave.keys())
    for var in allVariables:
        fileName = var + '.pt'
        torch.save(variablesToSave[var], subFolder + '/' + fileName)

def load(folderName, variablesToLoad):
    allVars = []
    for var in variablesToLoad:
        allVars.append(torch.load(folderName + '/' + var + '.pt'))
    return tuple(allVars)

def makeFolderOfTheDay(folderName):
    curDate = datetime.date.today().isoformat()
    folder = folderName + '/' + curDate
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return folder

def makeSubfolder(folder, project):
    curTime = datetime.datetime.now().strftime("%H_%M_%S")
    folder = folder + '/' + project + '_' + curTime
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return folder