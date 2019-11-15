# motiontrackingand-detection-opencv-java


AERIAL SURVEILLANCE AND MONITORING 
1. ABSTRACT 
With the advent of new technologies there’s a great threat to information and hence, security to a system plays a crucial role these days. Monitoring a large event, a protest, or even an individual, drones - unmanned aerial vehicles can provide the team with the overview they need to maintain control. UAV monitoring systems provide a number of benefits to users focused on public safety and civil security. This paper covers the capturing, detection and tracking the path of a drone and tracking by detection method used to track the multiple objects visual motion by detecting the objects in the frames and observing the tracklets throughout the entire frame and with the help of deep learning(subset of machine learning) techniques. This method gives high efficient tracking and considers longer term connectivity between pairs of detections and models similarities as well as dissimilarities between the objects position, color and visual motion. We here present the Hungarian method which gives a better performance and solves the problem of occlusion occurred between individuals. 
2. INTRODUCTION 
An unmanned aerial vehicle (commonly known as a drone) is an aircraft without a human pilot on board. UAVs are a component of an unmanned aircraft system (UAS) ; which include a UAV, a ground-based controller, and a system of communications between the two. These systems provide a risk free surveillance. 
Specially developed, highly efficient drone motors allow for discrete-almost silent- monitoring. Drone UAS are easy to use-No costly, time-consuming training programs are needed. Security pros can usually learn and implement missions in quite quickly. They are resistant to harsh weather like wind as well. They provide a payload flexibility i.e., monitoring systems can use a Convolutional video camera to detect suspects in darkness or dense vegetation. 
3. IMAGE PROCESSING 
Image processing is a method to convert an image into digital form and perform some operations on it, in order to get an enhanced image or to extract some useful information from it. Usually Image Processing system includes treating images as two dimensional signals while applying already set signal processing methods to them. The digital image processing deals with developing a digital system that performs operations on a digital image. 
What is an image? 
An image is nothing more than a two dimensional signal. It is defined by the mathematical function f(x,y) where x and y are the two co-ordinates horizontally and vertically. The value of f(x,y) at any point gives the pixel value at that point of an image. 
The dimensions of the picture is actually the dimensions of this two dimensional array. Machine vision or computer vision deals with developing a system in which the input is an image and the output is some information. For example: Developing a system that scans human face and opens any kind of lock. 
Pixel 
Pixel is the smallest element of an image. Each pixel corresponds to any one value. In an 8-bit gray scale image, the value of the pixel is between 0 and 255. The value of a pixel at any point corresponds to the intensity of the light photons striking at that point. Each pixel stores a value proportional to the light intensity at that particular location. 
4. OBJECT DETECTION Object detection is a computer technology related to computer vision and image processing that deals with detecting instances of semantic objects of a certain class (such as humans, buildings, or cars) in digital images and videos. There are various algorithms that can be used for object detection like RCNN, Fast RCNN and Faster RCNN, YOLO, SSD etc. each of which is discussed below. 
1. A Simple Way of Solving an Object Detection Task (using Deep Learning) 
The image below is a popular example illustrating how an object detection algorithm works. Each object in the image, from a person to a kite, have been located and identified with a certain level of precision. 
The simplest deep learning approach, and a widely used one, for detecting objects in images – Convolutional Neural Networks or CNNs. 
We pass an image to the network, and it is then sent through various convolutions and 
pooling layers. Finally, we get the output in the form of the object’s class. 
For each input image, we get a corresponding class as an output. We can use this technique to detect various objects in an image. Let’s look at how we can solve a general object detection problem using a CNN. 
1. First, we take an image as input: 
2. Then we divide the image into various regions: 
3. We will then consider each region as a separate image. 
4. Pass all these regions (images) to the CNN and classify them into various classes. 
5. Once we have divided each region into its corresponding class, we can combine all 
these regions to get the original image with the detected objects: 
The problem with using this approach is that the objects in the image can have different aspect ratios and spatial locations. For instance, in some cases the object might be covering most of the image, while in others the object might only be covering a small percentage of the image. The shapes of the objects might also be different. 
As a result of these factors, we would require a very large number of regions resulting in a huge amount of computational time. So to solve this problem and reduce the number of regions, we can use region- based CNN, which selects the regions using a proposal method. 
2. Region-Based Convolutional Neural Network 
Intuition of RCNN 
Instead of working on a massive number of regions, the RCNN algorithm proposes a bunch of boxes in the image and checks if any of these boxes contain any object. RCNN uses selective search to extract these boxes from an image (these boxes are called regions). 
There are basically four regions that form an object: varying scales, colors, textures, and enclosure. Selective search identifies these patterns in the image and based on that, proposes various regions. Here is a brief overview of how selective search works: 
• It first takes an image as input: 
• Then, it generates initial sub- segmentations so that, we have multiple regions from this image: 
• The technique then combines the similar regions to form a larger region (based on color similarity, texture similarity, size similarity, and shape compatibility): 
• Finally, these regions then produce the final object locations (Region of Interest). 
Below is a summary of the steps followed in RCNN to detect objects: 
Problems with RCNN 
We initially understood working of RCNN as helpful for object detection. But this technique comes with its own limitations. Training an RCNN model is expensive and slow with respect to the below steps: 
• CNN for feature extraction 
• Linear SVM classifier for identifying objects 

• Regression model for tightening the bounding boxes. 5. Background Subtraction 
As the name suggests, background subtraction is the process of separating out foreground objects from the background in a sequence of video frames. 
Fundamental Logic 
• Fundamental logic for detecting moving objects from the difference between the 

current frame and a reference frame, called “background image” and this method is known as FRAME DIFFERENCE METHOD. 
Background Modelling 
Background Model 
After Background Filtering... 
The screenshots of our work: 
Frame 1: Motion Tracking Frame 2: Background Subtraction Frame 3: Original Frame Frame 4: Multiple Motion Tracking 
Background Subtraction Principles: 

• Background subtraction should segment objects of interest when they first appear or reappear in a scene. 
• An appropriate pixel level stationery criterion should be defined. Pixels that satisfy this criterion are classified as background and ignored. 
• The background model must adapt to sudden and gradual changes in background. 
• Background model should take into account changes at differing spatial scales. 

The Problem – Requirements 
The background image is not fixed but must adapt to various changes like: 
Illumination changes 
Gradual 
Sudden (such as clouds) 
Motion changes: 
• Camera oscillations 
Changes in the background geometry: 
• Parked cars 
Simple Approach: I (x, y, t) B(x, y, t) 
1. Estimate the background for time t. 2. Subtract the estimated background from the input frame. 3. Apply a threshold, Th, to the absolute difference to get the foreground mask. 
The basic methods 
• Frame difference: 
| framei–framesi-1| > Th 
• The estimated background is just the previous frame 
• It evidently works only in particular conditions of objects’ speed and frame rate 
• Very sensitive to the threshold Th 
Frame Differencing 
• Background is estimated to be the previous Frame . Background subtraction equation then becomes: B(x ,y ,t) = I (x ,y ,t-1) I (x, y, t) - I (x, y, t – 1) > Th 
Algorithm Overview: 
6. PROPOSED TECHNIQUE 
The architecture of the proposed system the user inputs the video, where multi-object should be tracked. The video as it is cannot be processed. So video is converted into frame by frame. 
Every frame need not to be compared to detect the motion objects. It undergoes frame selection process, where some of the frames are skipped. This will be given as input to the Background model which produces the foreground object along with removal of noisy disturbance. Next step is to find the connected object in the foreground model. Blob analysis did this job by assigning unique label to every individual objects. And to predict the new location of the object in the next sequence frame, Kalman filters were used to make this process easier. Kalman filter is not actually the filter, they are the predictors. This predicts the new locations of objects. In case, if any of the noise is present at that stage it works like a filter and removes noisy disturbance. Predicted object has to be assigned to tracks, this is achieved by using Hungarian Algorithm, which uses matrix method to solve assignment detection to tracklets problem. 
7. IMPLEMENTATION: 
The stationary objects are getting separated from the motion object by using Gaussian mixture background subtraction model. Which when explained using probabilistic models, can be understood. First, the color model Bc and depth model Bd classify the observed pixels ct and dt at 

1. Object Detection 
Motion recognition is very important in automated surveillance system. It is the process of recognition of moving objects on the video source. Foreground object detection 

time t, respectively, pixel finds a matched Gaussian model using the Euclidian distance ||xt−μi|| ≥ k where k is a constant threshold identical to 2.5.We use the Euclidean distance to classify pixel since grayscale color and depth values are one dimension from 0 to255 in which there is no correlation. A pixel is classified as one of the following 3 cases. Case1: If a match is found and the pixel is classified as background, a matched model is the background model. Case2: If a match is found and the pixel is classified as foreground, a matched model is the foreground model. Case3: If no match is found, the pixel belongs to fore-ground. In the case of color value ct, after the pixels at time t are classified as background or foreground, the pixel values are computed by using the inequality for the matched Gaussian distribution as follows: θ·η(ct,μi,σi2)≥maximum pixel value in which θ is a constant to scale the Gaussian probability density function values. 
In the case 1, when a pixel is matched to a background model, if the inequality is satisfied, the pixel has 0. Otherwise, the pixel has θ·η. 
In the case 2, when a pixel is matched to a foreground model, if the inequality is satisfied, the pixel has 255. Otherwise, the pixel has 255-θ·η. 
In the case 3, the value becomes 255 because there is no a background or foreground model to decide the probability of the pixel. Eventually, the BGS results Rc consists of pixel values 0 to 255 where the higher probability P(ct)is closer to 0. 
The final BGS results R final are computed as follows if Rc(x, y) ∗Rd(x, y) > maximum pixel value then Rfinal(x, y) is foreground else then Rfinal(x, y) is background In this model, background pixels are label as 0 and foreground objects are label as 1.This is how the separation process is done, but the problem arises in the form of noisy disturbance. It can be solved by preprocessing, which uses morphological operations to remove it. Preprocessing 
Morphological operations such as erosion and dilation are used to remove noise from the pixels. 
It is a set of all points Z such that B, shifted or translated by Z, is contained in A. Erosion removes irrelevant size details from a binary image, shrinks and thins the image. It also strips away extrusions and strips apart joined objects. 
In Dialation, first B is reflected about its origin by a straight angle and then translated by Z. Dialation enalrges and bridges gaps of character by thickening and reparing breaking and intrusions. 
Blob Analysis For image processing, a blob is a region of connected pixels. After detecting foreground moving object and applying preprocessing operations, the filtered pixels are need to be grouped into a connected regions and are uniquely labeled. To solve this problem we use a simple two pass algorithm. On the first pass: 1. Iterate through each element of the data by row, then by column. 2. If the element is not the background 3. Get the neighboring elements of the current element 4. If there are no neighbors, uniquely label the current element and continue. 5. Otherwise, find the neighbor with the smallest label and assign it to the current element. 6. Store the equivalence between neighboring labels. 
On the second pass: 1. Iterate through each element of the data by column, then by row 2. If the element is not the background. Re- label the element with the lowest equivalent label. 
2. Object Tracking This section is to describe about the prediction and assignment to prediction process. 
Kalman Filter 
After the blobs (object) are detected the next step is to associate detection to the same object (track). This filter is used to predict the new detection in the consecutive frames and to associate those predictions to the location of a track in each frame. Kalman filter uses predict and correct method to do so and sometimes they also used as high- pass filters and low-pass filters to remove noisy disturbance. Hungarian Algorithm Assign detections to tracks in the process of tracking the multi-objects using James Munkers’s variant of the Hungarian assignment algorithm. It also determines which, are all the tracks that were missing and which detection should begin a new track. This method returns the indices of assigned tracks. Based on indices of assigned tracks the cost matrix decides the cost of the tracklets. If the cost is less, then the assignment is more and if the cost is more the assignment is less. In terms of mathematics the Euclidean metric is the distance between the two points in Euclidean space. Where the distance is smaller, then the assignment is more. Finally the tracking result is displayed to the user. 8. BACKGROUND SUBTRACTION 
ALGORITHMS: 
• The values of a particular pixel is modeled as a mixture of adaptive Gaussians. – Why mixture? Multiple surfaces appear in a pixel. – Why adaptive? Lighting conditions change. 
• At each iteration Gaussians are evaluated using a simple heuristic to determine which ones are mostly likely to correspond to the background. 
• Pixels that do not match with the “background Gaussians” are classified as foreground. 
• Foreground pixels are grouped using 2D connected component analysis. 
9. NOISE MODELS 
Noise tells unwanted information in digital images. Noise produces undesirable effects such as artifacts, unrealistic edges, unseen lines, corners, blurred objects and disturbs background scenes. To reduce these undesirable effects, prior learning of noise models is essential for further processing. 
Gaussian Noise Model 
It is also called as electronic noise because it arises in amplifiers or detectors. Gaussian noise caused by natural sources such as thermal vibration of atoms and discrete nature of radiation of warm objects [5]. Gaussian noise generally disturbs the gray values in digital images. That is why Gaussian noise model essentially designed and characteristics by its PDF or normalizes histogram with respect to gray value. This is given as 2 2 2 ( - ) - 1 2 ( ) 2 g P g e μ σ πσ = (1) Where g = gray value, σ = standard deviation and μ = mean. Generally Gaussian 
noise mathematical model represents the correct approximation of real world scenarios. In this noise model, the mean value is zero, variance is 0.1 and 256 gray levels in terms of its PDF. Due to this equal randomness the normalized Gaussian noise curve look like in bell shaped. The PDF of this noise model shows that 70% to 90% noisy pixel values of degraded image in between μ σ μ σ − + .The shape of normalized histogram is almost same in spectral domain. 
White Noise 
Noise is essentially identified by the noise power. Noise power spectrum is constant in white noise. This noise power is equivalent to power spectral density function. The statement “Gaussian noise is often white noise” is incorrect [4]. However neither Gaussian property implies the white sense. The range of total noise power is -∞ to +∞ available in white noise in frequency domain. That means ideally noise power is infinite in white noise. This fact is fully true because the light emits from the sun has all the frequency components. In white noise, 
correlation is not possible because of every pixel values are different from their neighbours. That is why autocorrelation is zero. So that image pixel values are normally disturb positively due to white noise. 10. Kalman filter 
A Kalman Filter is an algorithm that can predict future positions based on current position. It can also estimate current position better than what the sensor is telling us. It will be used to have better association. Kalman filter is a widely-used recursive technique for tracking linear dynamical systems under Gaussian noise. Many different versions have been proposed for background modeling, differing mainly in the state spaces used for tracking. 
This algorithm was basically developed for single dimensional and real valued signals which are associated with the linear systems assuming the system is corrupted with linear additive white Gaussian noise. 
The Kalman filter addresses the general problem of trying to estimate the state x ∈ Rn of a discrete-time controlled process that is governed by the linear difference equation xk = Axk – 1 + Buk – 1 + wk – 1 
•with a measurement z that is 
zk = Hxk + vk 
• The random variables wk and vk represent the process noise and measurement noise respectively. 
• The nxn matrix A in the previous difference equation relates the state at the previous time step k-1 to the state at the current step k, in the absence of either a driving function or process noise. 
• The nxl matrix B relates the optional control input u to the state x. 
• The mxn matrix H in the measurement equation relates the state to the measurement zk. 
The Computational Origins of the Filter: 
• We definex k−∈Rn to be our a priori state estimate at step k given knowledge of the process prior to step k , and x k∈Rn to be our a posteriori state estimate at step k given measurement zk. 
• We can then define a priori and a posteriori estimate errors as 
ek−≡ xk−x k− & 
ek≡ xk−x k 
The a priori estimate error covariance is then Pk−=Eek−ek−T 
•As such, the equations for the Kalman filter fall into two groups: time update equations and measurement update equations. 
•The time update equations are responsible for projecting forward (in time) the current state and error covariance estimates to obtain the a priori estimates for the next time step. 
The measurement update equations are responsible for the feedback—i.e. for incorporating a new measurement into the a priori estimate to obtain an improved a posteriori estimate. 
•The time update equations can also be thought of as predictor equations, while the measurement update equations can be thought of as corrector equations. 
•The final estimation algorithm resembles that of a predictor-corrector algorithm. 

• The a posteriori estimate error covariance is Pk=EekekT 
• The posteriori state estimate x k is written as a linear combination of an a priori estimate x k− and a weighted difference between an actual measurement zk & a measurement prediction H x k−. 
Kalman filter algorithm 
•The Kalman filter estimates a process by using a form of feedback control: the filter estimates the process state at some time and then obtains feedback in the form of (noisy) measurements. 

11. Hungarian Algorithm 
A Hungarian algorithm can tell if an object in current frame is the same as the one in previous frame. It will be used for association and id attribution. 
This algorithm can associate an obstacle from one frame to another, based on a score. We have many scores we can think of: 
• IOU (Intersection Over Union); meaning that if the bounding box is overlapping the previous one, it’s probably the same. 
• Shape Score; if the shape or size didn’t vary too much during two consecutives frames; the score increases. 
• Convolution Cost; we could run a CNN (Convolutional Neural Network) on the bounding box and compare this result with the one from a frame ago. If the convolutional 
features are the same, then it means the objects looks the same. If there is a partial occlusion, the convolutional features will stay partly the same and association will remain. 
1. We have two lists of boxes: a tracking list (t-1) and a detection list (t). 
2. Go through tracking and detection list, and calculate IOU, shape, convolutional score. For convolutions, cosine distance metrics would be used. 

Association: 
From frame a to frame b, we are tracking two obstacles (with id 1 and 2), adding one new detection (4) and keeping a track (3) in case it’s a false negative. 
The process for obtaining this is the following: 

3. In some cases of overlapping bounding boxes, we can have two or more matches for one candidate. 
We have a matrix that tells us matching between Detection and Trackings. The next thing is to call a function called linear_assignment() that implements the Hungarian Algorithm. This algorithm uses bipartite graph (graph theory) to find for each detection, the lowest tracking value in the matrix. Since we have scores and not costs, we will replace our 1 with -1; the minimum will be found. We can then check the values missing in Hungarian Matrix and consider them as unmatched detections, or unmatched trackings. 
12. CONCLUSION 
We have proposed a method for “video surveillance contexts “. There are so many different algorithms that do not give a perfect recognition and tracking of multi- object during the occlusion than this approach. The outcome of this paper will be a tracking of multi objects in the video which is inputted by the user and display the results. This system recognizes the multiple moving objects and tracking algorithm successfully tracks objects in consecutive frames. It also handles simple object occlusions. This model exploits the longer- term connectivities between pairs of detections. It also relies on pair wise similarities and dissimilarities factors defined at detection level, based on position, color and also visual motion cues. The model also incorporates a label for individual objects which make occlusion free. 
13. REFERENCES 
(1) S. Loncaric, A survey of shape analysis techniques. Pattern Recognition. (2) D. R. Magee, Tracking multiple vehicles using foreground, background and motion models. In Proc. of Statistical Methods in Video Processing. (3) A. M. McIvor, Background subtraction techniques. In Proc. Of Image and Vision Computing. (4)A. J. Lipton, H. Fujiyoshi, and R.S. Patil. Moving target classification and tracking from real-time video. In Proc. Of Workshop Applications of Computer Vision. (5)R. Cucchiara, M. Piccardi, and A. Prati, Detecting moving objects, ghosts, and shadows in video streams,” in the IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 24, October 2003, pp. 1337-1342. (6)L. Maddalena and A. Petrosino, A self- organizing approach to background subtraction for visual surveillance applications,” in the IEEE Transactions on Image Processing. (7)SeungJong Noh and Moongu Jeon, A new framework for background subtraction using multiple cues,” in Asian Conference on Computer Vision, 2012. 
