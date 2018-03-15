%load all data
data = load('fasterRCNNVehicleTrainingData.mat');
vehicleDataset = data.vehicleTrainingData;

%sample the data
vehicleDataset(1:4,:) %array with 2 columns - imageFilename, vehicle

%images are actually located in a separate directory
dataDir = fullfile(toolboxdir('vision'),'visiondata');

%modify the filename column to contain the complete directory path
vehicleDataset.imageFilename = fullfile(dataDir,vehicleDataset.imageFilename);

%%read in a sample image
%I = imread(vehicleDataset.imageFilename{10});
%%add the rectangles that represents the regions of interest
%I = insertShape(I,'Rectangle',vehicleDataset.vehicle{10});
%I=imresize(I,3);
%figure
%imshow(I)

I = imread(testData.imageFilename{10});
size(I)
%make training set
idx = floor(0.6*height(vehicleDataset));
trainingData = vehicleDataset(1:idx,:);
testData = vehicleDataset(idx:end,:);

%can load a pretrained model
%change this value to true if training a new model
doTrainingAndEval = false;

detector = data.detector;

    
%verify the training
I = imread('highway.png');

%run the trained detector on the loaded image
[bboxes,scores] = detect(detector,I);

%I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
%figure
%imshow(I)

%run the trained detector on each test image
resultsStruct = struct([]);
for i = 1:height(testData)
    I = imread(testData.imageFilename{i});
    [bboxes,scores,labels] = detect(detector,I);

    resultsStruct(i).Boxes = bboxes;
    resultsStruct(i).Scores = scores;
    resultsStruct(i).Labels = labels;
    
end
 
I = imread(testData.imageFilename{10});
I = insertObjectAnnotation(I,'rectangle',resultsStruct(10).Boxes,resultsStruct(10).Scores);
figure
imshow(I)
    
results = data.results;

%get the ground truch values for the evaluated objects
expectedResults = testData(:,2:end);

%evaluate the detector
[ap,recall,precision] = evaluateDetectionPrecision(results,expectedResults);