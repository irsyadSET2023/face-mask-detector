package ai.certifai.project;


import ai.certifai.handson.messyclean.MvcIterator;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;


public class TrainerFaceMask {


    private static int seed = 123;
    private static int numClasses = 2;
    private static int epoch = 10;
    private static int height = 224;
    private static int width = 224;
    private static int channel = 3;
    private static int batchSize = 12;
    public static final double trainingPercentage = 0.8;

    public static void main(String[] args) throws Exception {


        //Load Data

        File inputFile = new ClassPathResource("Dataset/test").getFile();

        MvcIterator iterator = new MvcIterator(inputFile, height, width, channel, batchSize, numClasses, trainingPercentage);

        DataSetIterator trainIter = iterator.getTrain();
        DataSetIterator testIter = iterator.getTest();

        //Input model VGG16

        ZooModel model = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) model.initPretrained();
        System.out.println(vgg16.summary());

        //model config

        FineTuneConfiguration ftConfig = new FineTuneConfiguration.Builder()
                .activation(Activation.RELU)
                .updater(new Adam(0.001))
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .l2(0.0005)
                .build();

        ComputationGraph vggTran = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(ftConfig)
                .setFeatureExtractor("fc1")
                .nOutReplace("fc1", 2048, WeightInit.XAVIER)
                .nInReplace("fc2", 2048, WeightInit.XAVIER)
                .nOutReplace("fc2", 64, WeightInit.XAVIER)
                .removeVertexKeepConnections("predictions")
                .addLayer("new-predictions", new OutputLayer.Builder()
                        .nIn(64)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build(),"fc2")
                .setOutputs("new-predictions")
                .build();

        System.out.println(vggTran.summary());

        //Model Fitting / Evaluation

        Evaluation evalTrain = vggTran.evaluate(trainIter);
        Evaluation evalTest = vggTran.evaluate(testIter);

        System.out.println("Train Evaluation" + evalTrain.stats());
        System.out.println("Test Evaluation" + evalTest.stats());

//        UIServer server = UIServer.getInstance();
//        StatsStorage statsStorage = new InMemoryStatsStorage();
//        server.attach(statsStorage);

        vggTran.setListeners(

                new ScoreIterationListener(10),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)

        );

        vggTran.fit(trainIter, epoch);
        System.out.println("Saving Model......");
        File outputFile = new File("dl4j-cv-labs/src/main/java/ai/certifai/project/mask detection model.zip");
        System.out.println(outputFile.getAbsolutePath());
        vggTran.save(outputFile);
        System.out.println("Model Training Ended");

    }
}
