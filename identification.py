import os
from sklearn.externals import joblib
import character_segmentation

# load the model
current_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_dir, 'models/svc/svc.pkl')
model = joblib.load(model_dir)

classification_result = []
for each_character in character_segmentation.characters:
    # to a 1D array
    each_character = each_character.reshape(1, -1);
    result = model.predict(each_character)
    classification_result.append(result)

print(classification_result)

plate_string = ''
for eachPredict in classification_result:
		  
    plate_string += eachPredict[0]

# sorting letters in correct order

column_list_copy = character_segmentation.column_list[:]
character_segmentation.column_list.sort()
rightplate_string = ''
for each in character_segmentation.column_list:
    rightplate_string += plate_string[column_list_copy.index(each)]

print(rightplate_string)
