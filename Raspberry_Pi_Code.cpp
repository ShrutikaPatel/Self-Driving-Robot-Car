// Note: I have considered BLACK Lane Lines on a WHITE floor !

// Include all the required libraries
#include <opencv2/opencv.hpp>
#include <raspicam_cv.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <wiringPi.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Image processing variable
using namespace std;
using namespace cv;
using namespace raspicam;

Mat frame, Matrix, framePers, frameGray,ROILane, frameFinalDuplicate, flattened_mat, frameInv;
Mat frameThresh, frameEdge, frameFinal,frameFinalDuplicate1, ROILaneEnd;
RaspiCam_Cv Camera;

vector<int> histogramLane; 
vector<int> histogramLaneEnd;

int LeftLanePos, RightLanePos, frameCenter, laneCenter,Result,laneEnd;

Point2f Source[] = {Point2f(45,160),Point2f(360,160),Point2f(0,210),Point2f(400,210)};            // Red Box ROI
Point2f Destination[] = {Point2f(100,0),Point2f(300,0),Point2f(100,240),Point2f(300,240)};        // Perspective Warp Box

stringstream ss;

// ML Variables 

CascadeClassifier Stop_Cascade, Object_Cascade, Traffic_Cascade; 
Mat frame_Stop, RoI_Stop, gray_Stop, frame_Object, RoI_Object, gray_Object, frame_Traffic, RoI_Traffic, gray_Traffic;
vector<Rect> Stop, Object, Traffic; 
int dist_Stop, dist_Object, dist_Traffic; 

 void Setup ( int argc,char **argv, RaspiCam_Cv &Camera )  // Camera Setup Function
  {
    Camera.set ( CAP_PROP_FRAME_WIDTH,  ( "-w",argc,argv,400 ) );
    Camera.set ( CAP_PROP_FRAME_HEIGHT,  ( "-h",argc,argv,240 ) );
    Camera.set ( CAP_PROP_BRIGHTNESS, ( "-br",argc,argv,50) );
    Camera.set ( CAP_PROP_CONTRAST ,( "-co",argc,argv,50) );
    Camera.set ( CAP_PROP_SATURATION,  ( "-sa",argc,argv,50 ) );
    Camera.set ( CAP_PROP_GAIN,  ( "-g",argc,argv ,50 ) );
    Camera.set ( CAP_PROP_FPS,  ( "-fps",argc,argv,0));

}

void Capture()            // 1st Function: For capturing the image / video. 
{
    Camera.grab();
    Camera.retrieve(frame);
    cvtColor(frame,frame,COLOR_BGR2RGB);        // Lane Detection frame
    cvtColor(frame, frame_Stop, COLOR_BGR2RGB); // Stop sign frame. 
    cvtColor(frame, frame_Object, COLOR_BGR2RGB); // Object frame. 
    cvtColor(frame, frame_Traffic, COLOR_BGR2RGB); // Traffic Light frame. 
}

void Perspective()  // 2nd Function: Perspective Wrapping (Bird-Eye View Transformation of the captured image/video.)
{
	Scalar white = Scalar(177, 201, 175);            // Set the boundary colour according to the floor colour on which the robot car will run. Tip: Use Color Picker!
	line(frame,Source[0],Source[1],white,2);
	line(frame,Source[1],Source[3],white,2); 
	line(frame,Source[3],Source[2],white,2);
	line(frame,Source[2],Source[0],white,2);  
	
	Matrix = getPerspectiveTransform(Source,Destination);
	warpPerspective(frame, framePers, Matrix, Size(400,240), INTER_LINEAR, BORDER_CONSTANT, white);
}

void Threshold()  // 3rd Function: Image Processing
{
	cvtColor(framePers,frameGray, COLOR_RGB2GRAY);
	bitwise_not(frameGray,frameInv);
	// adaptiveThreshold(frameGray, frameThresh, 255, ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, -10);  // Will work if the surrounding conditions are stable.
	inRange(frameInv, 170, 255, frameThresh);                                                               // only works for gray scale.  

	Canny(frameGray,frameEdge, 250, 500, 3, false); // 250, 500 Shrutika Home
	add(frameThresh, frameEdge, frameFinal);

	cvtColor(frameFinal,frameFinal,COLOR_GRAY2RGB);
	cvtColor(frameFinal,frameFinalDuplicate,COLOR_RGB2BGR);    // used in Histogram Only !
	cvtColor(frameFinal,frameFinalDuplicate1,COLOR_RGB2BGR);   // used in Histogram Only !
}


void Histogram()  // 4th function: Histogram Calculation for storing the pixel intensity of each coloun of the frame
{
	histogramLane.resize(400);
	histogramLane.clear();
	
	for(int i=0; i<frame.size().width; i++)
	{
		ROILane = frameFinalDuplicate(Rect(i,140,1,100));
		divide(255,ROILane, ROILane);
		histogramLane.push_back((int)(sum(ROILane)[0]));
	}
	
	/* histogramLaneEnd.resize(400);
	histogramLaneEnd.clear();
	
	for(int i=0; i<frame.size().width; i++)  // frame.size().width = 400
	{
		ROILaneEnd = frameFinalDuplicate1(Rect(i,0,1,240));
		divide(255,ROILaneEnd, ROILaneEnd);
		histogramLaneEnd.push_back((int)(sum(ROILaneEnd)[0]));
	}
	
	laneEnd = sum(histogramLaneEnd)[0];
	cout<<"Lane End"<<laneEnd<<endl; */
} 


void LaneFinder()  // 5th Function: For Finding the Lane Position based on function 4. 
{
	vector<int>:: iterator LeftPtr;
	LeftPtr = max_element(histogramLane.begin(), histogramLane.begin() + 180);
	LeftLanePos = distance(histogramLane.begin(), LeftPtr); 
	
	vector<int>:: iterator RightPtr;
	RightPtr = max_element(histogramLane.begin() + 220, histogramLane.end());
	RightLanePos = distance(histogramLane.begin(), RightPtr);
	
	line(frameFinal,Point2f(LeftLanePos,0), Point2f(LeftLanePos,240), Scalar(0,255,0),2);
	line(frameFinal,Point2f(RightLanePos,0), Point2f(RightLanePos,240), Scalar(0,255,0),2);

}

void LaneCenter()    // 6th function: Determining lane centre with the helf of above function and will consider a fixed frame centre.
{
	laneCenter = (RightLanePos-LeftLanePos)/2+LeftLanePos;
	frameCenter = 200;                                         // Will change with width of the frame.
	
	line(frameFinal, Point2f(laneCenter,0),Point2f(laneCenter,240),Scalar(0,255,0),3);
	line(frameFinal, Point2f(frameCenter,0),Point2f(frameCenter,240),Scalar(255,0,0),3);
	
	Result = laneCenter-frameCenter;
}

void Stop_detection()
{
    if(!Stop_Cascade.load("//home//pi//Desktop//MACHINE LEARNING//cascade_stop.xml"))   // file path.
	{ 
	   printf("Unable to open stop cascade file");
	}
	
	RoI_Stop = frame_Stop(Rect(200,0,200,140)); 
	cvtColor(RoI_Stop, gray_Stop, COLOR_RGB2GRAY);
	equalizeHist(gray_Stop, gray_Stop); 
	Stop_Cascade.detectMultiScale(gray_Stop, Stop);
	
	for(int i=0; i<Stop.size(); i++)                  // Creation of bounding box. 
	    {
		Point P1(Stop[i].x, Stop[i].y);
		Point P2(Stop[i].x + Stop[i].width, Stop[i].y + Stop[i].height);
		
		rectangle(RoI_Stop, P1, P2, Scalar(0, 0, 255), 2);
		putText(RoI_Stop, "Stop Sign", P1, FONT_HERSHEY_PLAIN, 1,  Scalar(0, 0, 255, 255), 2);
		dist_Stop = (-0.94)*(P2.x - P1.x) + 91.87; 
		
		ss.str(" ");
		ss.clear();
		ss<<"D="<<dist_Stop<< "cms" ;
		putText(RoI_Stop, ss.str(),Point2f(1,135),0,1,Scalar(0,0,255),2);
	    }
}

void Object_detection()
{
    if(!Object_Cascade.load("//home//pi//Desktop//MACHINE LEARNING//cascade_object.xml"))  // file path.
	{ 
	   printf("Unable to open Object cascade file");
	}
	
	RoI_Object = frame_Object(Rect(100,50,200,190)); 
	cvtColor(RoI_Object, gray_Object, COLOR_RGB2GRAY);
	equalizeHist(gray_Object, gray_Object); 
	Object_Cascade.detectMultiScale(gray_Object, Object);
	
	for(int i=0; i<Object.size(); i++)
	    {
		Point P1(Object[i].x, Object[i].y);
		Point P2(Object[i].x + Object[i].width, Object[i].y + Object[i].height);
		
		rectangle(RoI_Object, P1, P2, Scalar(0, 0, 255), 2);
		putText(RoI_Object, "Object!!", P1, FONT_HERSHEY_PLAIN, 1,  Scalar(0, 0, 255, 255), 2);
		dist_Object = (-0.576)*(P2.x - P1.x) + 77.3; 
		
		ss.str(" ");
		ss.clear();
		ss<<"D="<<dist_Object<< "cms" ;
		putText(RoI_Object, ss.str(),Point2f(1,150),0,1,Scalar(0,0,255),2);
	    }
}

void Traffic_detection()
{
    if(!Traffic_Cascade.load("//home//pi//Desktop//MACHINE LEARNING//cascade_traffic.xml"))  // file path.
	{ 
	   printf("Unable to open traffic cascade file");
	}
	
	RoI_Traffic = frame_Traffic(Rect(150,0,250,240)); 
	cvtColor(RoI_Traffic, gray_Traffic, COLOR_RGB2GRAY);
	equalizeHist(gray_Traffic, gray_Traffic); 
	Traffic_Cascade.detectMultiScale(gray_Traffic, Traffic);
	
	for(int i=0; i<Traffic.size(); i++)
	    {
		Point P1(Traffic[i].x, Traffic[i].y);
		Point P2(Traffic[i].x + Traffic[i].width, Traffic[i].y + Traffic[i].height);
		
		rectangle(RoI_Traffic, P1, P2, Scalar(0, 0, 255), 2);
		putText(RoI_Traffic, "Traffic Light", P1, FONT_HERSHEY_PLAIN, 1,  Scalar(0, 0, 255, 255), 2);
		dist_Traffic = (-1.153)*(P2.x - P1.x) + 94.615; 
		
		ss.str(" ");
		ss.clear();
		ss<<"D="<<dist_Traffic<< "cms" ;
		putText(RoI_Traffic, ss.str(),Point2f(1,135),0,1,Scalar(0,0,255),2);
	    }
}

int main(int argc,char **argv)
{
    
    wiringPiSetup();
    pinMode(21,OUTPUT);
    pinMode(22,OUTPUT);
    pinMode(23,OUTPUT);
    pinMode(24,OUTPUT);

    Setup(argc, argv, Camera);
    cout<<"Connecting to camera"<<endl;
    if (!Camera.open())
    {	
	cout<<"Failed to Connect"<<endl;
    }
     
    cout<<"Camera Id = "<<Camera.getId()<<endl;
    
    while(1)
    {
    auto start = std::chrono::system_clock::now();
    	
	// Calling all the functions. 

	Capture();
	Perspective();
	Threshold();
	Histogram();
	LaneFinder();
	LaneCenter();
	Stop_detection();
	Object_detection();
	Traffic_detection();
	
	if (dist_Stop > 20 && dist_Stop < 38)
	{
		digitalWrite(21,1);  // Decimal = 7
		digitalWrite(22,1);
		digitalWrite(23,1);
		digitalWrite(24,0);
		cout<<"Stop Sign"<<endl; 
		dist_Stop = 0;
		goto Stop_Sign; 
	}
	
	if (dist_Object > 5 && dist_Object < 25)
	{
		digitalWrite(21,0);  // Decimal = 8
		digitalWrite(22,0);
		digitalWrite(23,0);
		digitalWrite(24,1);
		cout<<"Object"<<endl; 
		dist_Object = 0;
		goto Object; 
	}
	
		
	if (dist_Traffic > 40 && dist_Traffic < 50)
	{
		digitalWrite(21,1);  // Decimal = 9
		digitalWrite(22,0);
		digitalWrite(23,0);
		digitalWrite(24,1);
		cout<<"Traffic"<<endl; 
		dist_Traffic = 0;
		goto Traffic; 
	}
	
	if(Result == 0)
	{
		digitalWrite(21,0);  // Decimal = 0
		digitalWrite(22,0);
		digitalWrite(23,0);
		digitalWrite(24,0);
		cout<<"Forward"<<endl;
	}
	else if(Result >=1 && Result < 5)
	{
		digitalWrite(21,1);  // Decimal = 1
		digitalWrite(22,0);
		digitalWrite(23,0);
		digitalWrite(24,0);
		cout<<"Right1"<<endl;
	}
	else if(Result >= 5 && Result < 10)
	{
		digitalWrite(21,0);  // Decimal = 2
		digitalWrite(22,1);
		digitalWrite(23,0);
		digitalWrite(24,0);
		cout<<"Right2"<<endl;
	}
	else if(Result >= 10)
	{
		digitalWrite(21,1);  // Decimal = 3
		digitalWrite(22,1);
		digitalWrite(23,0);
		digitalWrite(24,0);
		cout<<"Right3"<<endl;
	}
	
	else if(Result <= -1 && Result > -5)
	{
		digitalWrite(21,0);  // Decimal = 4
		digitalWrite(22,0);
		digitalWrite(23,1);
		digitalWrite(24,0);
		cout<<"Left1"<<endl;
	}
	else if(Result <= -5 && Result > -10)
	{
		digitalWrite(21,1);  // Decimal = 5
		digitalWrite(22,0);
		digitalWrite(23,1);
		digitalWrite(24,0);
		cout<<"Left2"<<endl;
	}
	else if(Result <= -10)
	{
		digitalWrite(21,0);  // Decimal = 6
		digitalWrite(22,1);
		digitalWrite(23,1);
		digitalWrite(24,0);
		cout<<"Left3"<<endl;
	} 
	
	Stop_Sign:
	Object: 
	Traffic:
		
	if(Result == 0)
	{
	ss.str(" ");
	ss.clear();
	ss<<"Result="<<Result<<"(Move Forward)";
	putText(frame, ss.str(),Point2f(1,35),0,1,Scalar(0,0,255),2);
    } 
    
    else if(Result > 0)
	{
	ss.str(" ");
	ss.clear();
	ss<<"Result="<<Result<<"(Move Right)";
	putText(frame, ss.str(),Point2f(1,35),0,1,Scalar(0,0,255),2);
    }
    
    else if(Result < 0)
	{
	ss.str(" ");
	ss.clear();
	ss<<"Result="<<Result<<"(Move Left)";
	putText(frame, ss.str(),Point2f(1,35),0,1,Scalar(0,0,255),2);
    }
	
	// Move windows for displaying results according to your requirements. 
	
	namedWindow("Original Image", WINDOW_KEEPRATIO);
	moveWindow("Original Image", 5,90);
	resizeWindow("Original Image",400, 240);
	imshow("Original Image", frame);
	
	//namedWindow("Perspective Wrap", WINDOW_KEEPRATIO);
	//moveWindow("Perspective Wrap", 420,90);
	//resizeWindow("Perspective Wrap",400, 240);
	//imshow("Perspective Wrap", framePers); 
	
	namedWindow("Final Image", WINDOW_KEEPRATIO);
	moveWindow("Final Image", 830,90);
	resizeWindow("Final Image",400, 240);
	imshow("Final Image", frameFinal);
	
	//namedWindow("Edges", WINDOW_KEEPRATIO);
	//moveWindow("Edges", 830,410);
	//resizeWindow("Edges",400, 240);
	//imshow("Edges", frameEdge);  
	
	//namedWindow("Inverted Image", WINDOW_KEEPRATIO);
	//moveWindow("Inverted Image", 5,340);
	//resizeWindow("Inverted Image",400, 240);
	//imshow("Inverted Image", frameInv);
	
	namedWindow("Thresholded Image", WINDOW_KEEPRATIO);
	moveWindow("Thresholded Image", 420,90);
	resizeWindow("Thresholded Image",400, 240);
	imshow("Thresholded Image", frameThresh);
	
	namedWindow("Stop Sign", WINDOW_KEEPRATIO);
	moveWindow("Stop Sign", 5,370);
	resizeWindow("Stop Sign",400, 240);
	imshow("Stop Sign", RoI_Stop);
	
	namedWindow("Object Detection", WINDOW_KEEPRATIO);
	moveWindow("Object Detection", 415,370);
	resizeWindow("Object Detection",400, 240);
	imshow("Object Detection", RoI_Object); 
	

	namedWindow("Traffic", WINDOW_KEEPRATIO);
	moveWindow("Traffic", 820,370);
	resizeWindow("Traffic",400, 240);
	imshow("Traffic", RoI_Traffic);
	
	waitKey(1);
	
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    
    float t = elapsed_seconds.count();
    int FPS = 1/t;
    cout<<"FPS = "<<FPS<<endl;
    cout<<"Result = "<<Result<<endl;
    }
    return 0;   
}
