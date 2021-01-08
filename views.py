from flask import Flask, render_template, redirect, url_for, request,session
from searcher import Searcher
from colorDescriptor import RGBHistogram
from LBP import LocalBinaryPatterns
import cv2
import numpy as np
import os
import pickle

#specify where our data will reside 
UPLOAD_FOLEDR = "./static/uploads"
DATA_FOLDER = "./static/data"
                                                                                                            

app = Flask(__name__)
app.secret_key = 'mydata'
#displaying our home page
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload_image", methods=["GET", "POST"])
def uploadImage() :
    if request.method =='POST':
        images ={}
        if request.files:
            img = request.files['uploadImage']
            imgName = img.filename
            imagePath = os.path.join(UPLOAD_FOLEDR,imgName)
            img.save(imagePath)
            print(imagePath)
            images = search(imagePath) 
            session['images'] = images 
        return  redirect(url_for('displayResult'))


def search(queryImagePath):
    # load the query image and show it
    queryImage=cv2.imread(queryImagePath)

    #cv2.imshow("Query", queryImage)
    print("query: {}".format(queryImagePath))

    # extract the feature vectors using RGBHIstogram and Local  ninary Pattern for the texture
    desc_text = LocalBinaryPatterns(24,8)
    desc_color = RGBHistogram([8,8,8])                                                                                                                                                                                                                                                                      
    queryFeatures_text = desc_text.describe(queryImage)
    queryFeatures_color = desc_color.describe(queryImage)
    # load the index and initialize our searchers
    index_text = pickle.loads(open("textureIndex", "rb").read())
    index_color = pickle.loads(open("colorIndex", "rb").read())
    searcher_text = Searcher(index_text)
    searcher_color = Searcher(index_color)
    #then we perform the search
    results_text = searcher_text.search(queryFeatures_text)
    results_color = searcher_color.search(queryFeatures_color)                                                                             
    
    paths_text ={}
    paths_color ={}
    
    # loop over the  top 100 images that apears in all descriptors results
    for j in range(0, 100):
        # grab the result  and
        # load the result image
        (score, imageName) = results_text[j]
        path = os.path.join(DATA_FOLDER, imageName)
        print(path)
        paths_text[path] = score
        
        (score, imageName) = results_color[j]
        path = os.path.join(DATA_FOLDER, imageName)
        print(path)
        paths_color[path] = score
        print("\t{}. {} : {:.3f}".format(j + 1, imageName, score))
    paths = mergeDict(paths_text,paths_color)  
    
    '''
    for j in range(0,10) :
        (score, imageName) = results_text[j] if results_text[j][0]< results                                                                                                éé_color[j][0] else results_color[j]
        path = os.path.join(DATA_FOLDER, imageName)
        print(path)
        paths[path] = score  
        '''                                                                            
    return paths
 

@app.route("/Result")
def displayResult() :
    images = session['images']
    return render_template("result.html", data=images)
   
def mergeDict(dict1, dict2):
    dict3 = {x:[dict1[x],dict2[x]] for x in dict1  
                              if x in dict2}
    return dict3


if __name__ == "__main__" :
    app.run()
