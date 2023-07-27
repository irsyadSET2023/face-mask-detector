package ai.certifai.project;

import ai.certifai.solution.facial_recognition.FaceRecognitionWebcam;
import ai.certifai.solution.facial_recognition.detection.FaceDetector;
import ai.certifai.solution.facial_recognition.detection.FaceLocalization;
import ai.certifai.solution.facial_recognition.detection.OpenCV_DeepLearningFaceDetector;
import ai.certifai.solution.facial_recognition.detection.OpenCV_HaarCascadeFaceDetector;
import ai.certifai.solution.facial_recognition.identification.DistanceFaceIdentifier;
import ai.certifai.solution.facial_recognition.identification.FaceIdentifier;
import ai.certifai.solution.facial_recognition.identification.Prediction;
import ai.certifai.solution.facial_recognition.identification.feature.InceptionResNetFeatureProvider;
import ai.certifai.solution.facial_recognition.identification.feature.VGG16FeatureProvider;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.flip;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_highgui.destroyAllWindows;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_HEIGHT;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_PROP_FRAME_WIDTH;

public class MaskCameraDetector {

    private static final Logger log = LoggerFactory.getLogger(FaceRecognitionWebcam.class);
    private static final int WIDTH = 1280;
    private static final int HEIGHT = 720;
    private static final String outputWindowsName = "Mask Camera Detector";
    public static final  String MODEL_FILENAME = "C:\\Users\\ASUS\\Downloads\\mask detection model.zip";

    public static void main(String[] args) throws Exception{

        System.out.println("Loading Model......");
        File modelFile = new File(MODEL_FILENAME);
        ComputationGraph model = ModelSerializer.restoreComputationGraph(modelFile,true);

        //        STEP 1 : Select your face detector and face identifier
        //        You can switch between different FaceDetector and FaceIdentifier options to test its performance
        FaceDetector FaceDetector = getFaceDetector(ai.certifai.solution.facial_recognition.detection.FaceDetector.OPENCV_DL_FACEDETECTOR);

        //        STEP 2 : Stream the video frame from camera
        VideoCapture capture = new VideoCapture();
        capture.set(CAP_PROP_FRAME_WIDTH, WIDTH);
        capture.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);
        namedWindow(outputWindowsName, WINDOW_NORMAL);
        resizeWindow(outputWindowsName, 1280, 720);

        if (!capture.open(0)) {
            System.out.println("Cannot open the camera !!!");
        }

        Mat image = new Mat();
        Mat cloneCopy = new Mat();

        while (capture.read(image)) {
            flip(image, image, 1);

            //        STEP 3 : Perform face detection
            image.copyTo(cloneCopy);
            FaceDetector.detectFaces(cloneCopy);
            List<FaceLocalization> faceLocalizations = FaceDetector.getFaceLocalization();


            //        STEP 4 : Perform face recognition
            INDArray[] result = predict(model, image);

            double[] resultVector = result[0].toDoubleVector();
            double prd=resultVector[0];

            if(prd>0.7){
              String mask="Wearing Mask";
                annotateFaces1(faceLocalizations, image,mask);
            }

            else{
                String mask="Not Wearing Mask";
                annotateFaces2(faceLocalizations, image,mask);
            }



            image.copyTo(cloneCopy);


            //        STEP 5 : Display output in a window
            imshow(outputWindowsName, image);

            char key = (char) waitKey(20);
            // Exit this loop on escape:
            if (key == 27) {
                destroyAllWindows();
                break;
            }
        }
    }

    private static FaceDetector getFaceDetector(String faceDetector) throws IOException {
        switch (faceDetector) {
            case FaceDetector.OPENCV_HAAR_CASCADE_FACEDETECTOR:
                return new OpenCV_HaarCascadeFaceDetector();
            case FaceDetector.OPENCV_DL_FACEDETECTOR:
                return new OpenCV_DeepLearningFaceDetector(300, 300, 0.8);
            default:
                return  null;
        }
    }

    //    Method to draw the predicted bounding box of the detected face
    private static void annotateFaces1(List<FaceLocalization> faceLocalizations, Mat image,String mask) {
        for (FaceLocalization i : faceLocalizations){
            rectangle(image,new Rect(new Point((int) i.getLeft_x(),(int) i.getLeft_y()), new Point((int) i.getRight_x(),(int) i.getRight_y())), new Scalar(0, 255, 0, 0),2,8,0);
            putText(image, mask,new Point((int) i.getLeft_x()-1, (int) i.getLeft_y()-1), 3, 0.5, new Scalar(255,255,255,50));
        }

    }

    private static INDArray[] predict(ComputationGraph model, Mat image) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(224,224,3);
        INDArray image2 = loader.asMatrix(image);
        DataNormalization scaler = new ImagePreProcessingScaler();
        scaler.transform(image2);
        INDArray[] result = model.output(false,image2);
        return result;
    }

    private static void annotateFaces2(List<FaceLocalization> faceLocalizations, Mat image,String mask) {
        for (FaceLocalization i : faceLocalizations){
            rectangle(image,new Rect(new Point((int) i.getLeft_x(),(int) i.getLeft_y()), new Point((int) i.getRight_x(),(int) i.getRight_y())), new Scalar(0, 0, 255, 0),2,8,0);
            putText(image, mask,new Point((int) i.getLeft_x()-1, (int) i.getLeft_y()-1), 3, 0.5, new Scalar(255,255,255,50));
        }

    }


}
