from fiji.plugin.trackmate import Model
from ij import WindowManager
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import DetectorKeys
from fiji.plugin.trackmate.detection import DownsampleLogDetectorFactory
from fiji.plugin.trackmate.tracking.sparselap import SparseLAPTrackerFactory
from ij import IJ, WindowManager
from fiji.plugin.trackmate.tracking import LAPUtils
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter
import sys
import csv
import shutil 
import os
import fiji.plugin.trackmate.features.track.TrackDurationAnalyzer as TrackDurationAnalyzer
import fiji.plugin.trackmate.features.track.TrackSpeedStatisticsAnalyzer as TrackSpeedStatisticsAnalyzer

import os
from ij import IJ, ImagePlus
from ij.gui import GenericDialog
from ij.measure import Calibration

import java.io.File as File
import java.util.ArrayList as ArrayList



def run():
  print "run:"

  srcDir = IJ.getDirectory("Input_directory")
  if not srcDir:
    return
    print "Error1:"
  dstDir = IJ.getDirectory("Output_directory")
  if not dstDir:
    return
    print "Error2:"

  for root, directories, filenames in os.walk(srcDir):
    for filename in filenames:
      # Check for file extension
      if not filename.endswith(ext):
        continue
      # Check for file name pattern
      if containString not in filename:
        continue
      process(srcDir, dstDir, root, filename, keepDirectories)


def process(srcDir, dstDir, currentDir, fileName, keepDirectories):
  print "Processing:"
     
  # Opening the image
  print "Open image file", fileName
  imp = IJ.openImage(os.path.join(currentDir, fileName))

    

  #Here we make sure the calibration are correct
  units = "micron"
  TimeUnit = "sec"

  newCal = Calibration()
  newCal.pixelWidth = Pixel_calibration
  newCal.pixelHeight = Pixel_calibration
  newCal.frameInterval = Time_interval
  
  newCal.setXUnit(units)
  newCal.setYUnit(units)
  newCal.setTimeUnit(TimeUnit)
  imp.setCalibration(newCal)
  cal = imp.getCalibration()

  dims = imp.getDimensions() # default order: XYCZT

  if (dims[4] == 1):
  	imp.setDimensions(1, 1, dims[3]) 


  
  #imp.setDimensions(dim[2], dim[4], dim[3] )
  
 

 # Start the tracking
       
  model = Model()

  #Read the image calibration
  model.setPhysicalUnits( cal.getUnit(), cal.getTimeUnit() )
    	
# Send all messages to ImageJ log window.
  model.setLogger(Logger.IJ_LOGGER)
  
  settings = Settings()
  settings.setFrom(imp)
       
# Configure detector - We use the Strings for the keys
# Configure detector - We use the Strings for the keys
  settings.detectorFactory = DownsampleLogDetectorFactory()
  settings.detectorSettings = {
  DetectorKeys.KEY_RADIUS: 2.,
  DetectorKeys.KEY_DOWNSAMPLE_FACTOR: 2,
  DetectorKeys.KEY_THRESHOLD : 1.,}
		
  print(settings.detectorSettings)

    
# Configure spot filters - Classical filter on quality
  filter1 = FeatureFilter('QUALITY', 0, True)
  settings.addSpotFilter(filter1)
     
# Configure tracker - We want to allow merges and fusions
  settings.trackerFactory = SparseLAPTrackerFactory()
  settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap() # almost good enough
  settings.trackerSettings['LINKING_MAX_DISTANCE'] = LINKING_MAX_DISTANCE
  settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = ALLOW_TRACK_SPLITTING
  settings.trackerSettings['SPLITTING_MAX_DISTANCE'] = SPLITTING_MAX_DISTANCE
  settings.trackerSettings['ALLOW_TRACK_MERGING'] = ALLOW_TRACK_MERGING
  settings.trackerSettings['MERGING_MAX_DISTANCE'] = MERGING_MAX_DISTANCE

   
# Configure track analyzers - Later on we want to filter out tracks 
# based on their displacement, so we need to state that we want 
# track displacement to be calculated. By default, out of the GUI, 
# not features are calculated. 
    
# The displacement feature is provided by the TrackDurationAnalyzer.
    
  settings.addTrackAnalyzer(TrackDurationAnalyzer())
  settings.addTrackAnalyzer(TrackSpeedStatisticsAnalyzer())

  filter2 = FeatureFilter('TRACK_DISPLACEMENT', 10, True)
  settings.addTrackFilter(filter2)
    
    
#-------------------
# Instantiate plugin
#-------------------
  trackmate = TrackMate(model, settings)
       
#--------
# Process
#--------
    
  ok = trackmate.checkInput()
  if not ok:
  	sys.exit(str(trackmate.getErrorMessage()))
    
  ok = trackmate.process()
  if not ok:
  	sys.exit(str(trackmate.getErrorMessage()))
    
       
#----------------
# Display results
#----------------
  if showtracks: 
  	model.getLogger().log('Found ' + str(model.getTrackModel().nTracks(True)) + ' tracks.')
  	selectionModel = SelectionModel(model)
  	displayer =  HyperStackDisplayer(model, selectionModel, imp)
  	displayer.render()
  	displayer.refresh()
   
# The feature model, that stores edge and track features.
  fm = model.getFeatureModel()
        
  with open(dstDir+fileName+'tracks_properties.csv', "w") as file:
  	writer1 = csv.writer(file)
  	writer1.writerow(["track #","TRACK_MEAN_SPEED", "TRACK_MAX_SPEED", "NUMBER_SPLITS", "TRACK_DURATION", "TRACK_DISPLACEMENT" ])

  	with open(dstDir+fileName+'spots_properties.csv', "w") as trackfile:
  		writer2 = csv.writer(trackfile)
  		#writer2.writerow(["spot ID","POSITION_X","POSITION_Y","Track ID", "FRAME"])
  		writer2.writerow(["Tracking ID","Timepoint","Time (secs)","X pos", "Y pos"])

  		for id in model.getTrackModel().trackIDs(True):
   
   	 # Fetch the track feature from the feature model.
  			v = fm.getTrackFeature(id, 'TRACK_MEAN_SPEED')
  			ms = fm.getTrackFeature(id, 'TRACK_MAX_SPEED')
  			s = fm.getTrackFeature(id, 'NUMBER_SPLITS')
  			d = fm.getTrackFeature(id, 'TRACK_DURATION')
  			e = fm.getTrackFeature(id, 'TRACK_DISPLACEMENT')
  			model.getLogger().log('')
  			model.getLogger().log('Track ' + str(id) + ': mean velocity = ' + str(v) + ' ' + model.getSpaceUnits() + '/' + model.getTimeUnits())
       
  			track = model.getTrackModel().trackSpots(id)
  			writer1.writerow([str(id),str(v),str(ms),str(s),str(d),str(e)])
  		  	  		    
  			for spot in track:
  				sid = spot.ID()
  				x=spot.getFeature('POSITION_X')
  				y=spot.getFeature('POSITION_Y')
  				z=spot.getFeature('TRACK_ID')
  				t=spot.getFeature('FRAME')
  				time= int(t) * int(Time_interval)
  				writer2.writerow([str(id), str(t), str(time), str(x), str(y)])
  				#writer2.writerow([str(sid), str(x), str(y), str(id), str(t)])


gd = GenericDialog("Tracking settings")
gd.addNumericField("Pixel_calibration (micron)", 0.63, 0)
gd.addNumericField("Time Frame (s)", 300, 1)
gd.addNumericField("LINKING_MAX_DISTANCE", 20, 1)
gd.addCheckbox("ALLOW_TRACK_SPLITTING", True)
gd.addNumericField("SPLITTING_MAX_DISTANCE", 20, 1)
gd.addCheckbox("ALLOW_TRACK_MERGING", False)
gd.addNumericField("MERGING_MAX_DISTANCE", 20, 1)
gd.addStringField("File_extension", ".tif")
gd.addStringField("File_name_contains", "")
gd.addCheckbox("Keep directory structure when saving", True)
gd.addCheckbox("Show tracks", False)  
gd.showDialog()

Pixel_calibration = gd.getNextNumber()
Time_interval = gd.getNextNumber()
LINKING_MAX_DISTANCE = gd.getNextNumber()
ALLOW_TRACK_SPLITTING = gd.getNextBoolean()
SPLITTING_MAX_DISTANCE = gd.getNextNumber()
ALLOW_TRACK_MERGING = gd.getNextBoolean()
MERGING_MAX_DISTANCE = gd.getNextNumber()  
ext = gd.getNextString()
containString = gd.getNextString()
keepDirectories = gd.getNextBoolean()
showtracks = gd.getNextBoolean()

#print(LINKING_MAX_DISTANCE)
#print(ALLOW_TRACK_SPLITTING)
#print(SPLITTING_MAX_DISTANCE)
#print(ALLOW_TRACK_MERGING)
#print(MERGING_MAX_DISTANCE)
#print(LINKING_MAX_DISTANCE)
#print(ext)

print(Pixel_calibration)
print(Time_interval)

run()
