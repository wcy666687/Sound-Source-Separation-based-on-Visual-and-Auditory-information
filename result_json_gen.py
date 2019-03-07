import json
import os
import feat_extractor
import data_create
def jsonGen(filepath,OutputPath):
	filelist=os.listdir(filepath)
	result={}
	genres = ['accordion', 'acoustic_guitar', 'cello', 'flute', 'saxophone', 'trumpet', 'violin', 'xylophone']
	for file in filelist:
		name1,name2=feat_extractor.test('testimage\\'+file)
		flag=0
		if (name1 in file) and (name2 in file):
			if genres.index(name1)<genres.index(name1):
				flag=1
			else:
				flag=2
		else:
			x,y=data_create.type_verify(file)
			if name1 in file:
				if name1==x:
					flag=1
				else:
					flag=2
			if name1 in file:
				if name2==x:
					flag=2
				else:
					flag=1


		file_key=file+'.mp4'
		result[file_key]=[]
		temp={}
		temp['audio']=file+'_seg1.wav'
		if flag==1:
			temp['position']=0
		else:
			temp['position'] = 1

		result[file_key].append(temp)
		temp={}
		temp['audio']=file+'_seg2.wav'
		if flag==1:
			temp['position']=1
		else:
			temp['position'] = 0
		result[file_key].append(temp)

	with open(os.path.join(OutputPath,"result.json"),"w") as f:
		json.dump(result,f,indent=4)

if __name__ == '__main__':
	filepath='./testimage'
	OutputPath="./result_json"
	if not os.path.exists(OutputPath):
		os.mkdir(OutputPath)
	jsonGen(filepath,OutputPath)