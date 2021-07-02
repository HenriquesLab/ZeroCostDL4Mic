SaveDirectory = "D:/Desktop"

n = roiManager("count");
id = getImageID();
selectImage(id);
title = getTitle();
width = getWidth();
height = getHeight();
print("filename,width,height,class,index,x,y");
for (index = 0; index < n; index++) {
    roiManager("select", index);
    getSelectionCoordinates(x, y);
    for (i = 0; i < x.length; i++) {
        print(title+","+width+","+height+","+Roi.getName+","+index+1+","+round(x[i])+","+round(y[i]));
    }
} 

string = getInfo("log");

File.saveString(string, SaveDirectory+"/"+title+".csv");
roiManager("reset");
run("Close");
selectImage(id);
run("Close");