import json
import os

def jsonGen(filepath,OutputPath):
	filelist=os.listdir(filepath)
	result={}
	for file in filelist:
		file_key=file+'.mp4'
		result[file_key]=[]
		temp={}
		temp['audio']=file+'_gt1.wav'
		temp['position']=0
		result[file_key].append(temp)
		temp={}
		temp['audio']=file+'_gt2.wav'
		temp['position']=1
		result[file_key].append(temp)

	with open(os.path.join(OutputPath,"gt.json"),"w") as f:
		json.dump(result,f,indent=4)

if __name__ == '__main__':
	filepath='./testimage'
	OutputPath="./result_json"
	if not os.path.exists(OutputPath):
		os.mkdir(OutputPath)
	jsonGen(filepath,OutputPath)