#if defined(__linux__) || defined(LINUX) || defined(__APPLE__) || defined(ANDROID) || (defined(_MSC_VER) && _MSC_VER>=1800)

#include <opencv2/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d.hpp>
#include <opencv2/objdetect.hpp>

#include <raspicam_cv.h>

#include <stdio.h>

using namespace std;
using namespace cv;

const string WindowName = "Face Detection example";

class CascadeDetectorAdapter : public DetectionBasedTracker::IDetector
{
public:
	CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector)
		: IDetector()
		, Detector(detector)
	{
		CV_Assert(detector);
	}

	void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects)
	{
		Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
	}

	virtual ~CascadeDetectorAdapter()
	{}

private:
	CascadeDetectorAdapter();
	cv::Ptr<cv::CascadeClassifier> Detector;
};

int main(int, char**)
{
	namedWindow(WindowName, WINDOW_AUTOSIZE);	
	
	raspicam::RaspiCam_Cv cam;
	cam.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cam.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	cam.set(CV_CAP_PROP_FORMAT,CV_8UC3);
	cam.set(CV_CAP_PROP_FPS, 25);
	
	cam.set(CV_CAP_PROP_EXPOSURE, 100);
	const int frameDelay = 10;
	
	if (!cam.open())
		return 1;
	
	std::string cascadeFrontalfilename = "/share/OpenCV/lbpcascades/lbpcascade_frontalface_improved.xml";
	cv::Ptr<cv::CascadeClassifier> cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);
	cv::Ptr<DetectionBasedTracker::IDetector> MainDetector = makePtr<CascadeDetectorAdapter>(cascade);
	if (cascade->empty())
	{
		printf("Error: Cannot load %s\n", cascadeFrontalfilename.c_str());
		return 2;
	}

	cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);
	cv::Ptr<DetectionBasedTracker::IDetector> TrackingDetector = makePtr<CascadeDetectorAdapter>(cascade);
	if (cascade->empty())
	{
		printf("Error: Cannot load %s\n", cascadeFrontalfilename.c_str());
		return 2;
	}

	DetectionBasedTracker::Parameters params;
	DetectionBasedTracker Detector(MainDetector, TrackingDetector, params);

	if (!Detector.run())
	{
		printf("Error: Detector initialization failed\n");
		return 2;
	}

	Mat ReferenceFrame;
	Mat GrayFrame;
	vector<Rect> Faces;

	do
	{
		RNG rng(12345);
		cam.grab();
		cam.retrieve(ReferenceFrame);
		cvtColor(ReferenceFrame, GrayFrame, COLOR_BGR2GRAY);
		Detector.process(GrayFrame);
		Detector.getObjects(Faces);

		for (size_t i = 0; i < Faces.size(); i++)
		{
			rectangle(ReferenceFrame, Faces[i], Scalar(0, 255, 0));
		}

		imshow(WindowName, ReferenceFrame);
	} while (waitKey(frameDelay) < 0);

	cam.release();
	Detector.stop();

	return 0;
}

#else

#include <stdio.h>
int main()
{
	printf("This sample works for UNIX or ANDROID or Visual Studio 2013+ only\n");
	return 0;
}

#endif
