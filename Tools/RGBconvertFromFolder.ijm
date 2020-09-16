// This macro converts all images in a folder into RGB images and saves them into a folder called RGB files
// Romain F. Laine 2020-09-16
// r.laine@ucl.ac.uk


dir = getDirectory("Choose a Directory ");

folder_save = dir+File.separator+"RGB files";
File.makeDirectory(folder_save);

print("\\Clear");

setBatchMode("hide");
list = getFileList(dir);
for (i=0; i<list.length; i++) {
	if (!endsWith(list[i], "/")){
		print(list[i]);
		open(dir+list[i]);
		run("RGB Color");
		saveAs("PNG", folder_save+File.separator+list[i]);
		close(list[i]);
	}
}

setBatchMode("exit and display");
print("---------");
print("All done.");

