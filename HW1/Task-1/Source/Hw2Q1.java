import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.logging.Logger;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

public class Hw2Q1 {

	private static final Logger logger = Logger.getLogger("DataSet");

	public static class DataSet {
		public Instances trainingSet; 
		public Instances testSet;
	}

	/**
	 * Main method. Do not change inside of this function.
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		float learningRates[] = { 1f, 0.5f, 0.1f, 0.01f, 0.001f };
		int classIndex = 8;
		String hiddenLayers[] = { "10", "10,15", "10,15,25" };
		float momentum = 0.2f;
		int maxEpoch = 5000;

		if (args.length!=1) {
			// logger.error("You should give the path of the dataset!");
			return;
		}
		Instances data;
		try {
			data = readData(args[0]);
			data.setClassIndex(classIndex);

			DataSet dataSet = splitData(data, 0.8f, false);

			for (String hiddenLayer : hiddenLayers) {

				for (float learningRate : learningRates) {
					logger.info(String.format(
							"\nMLP with arguments:\n==========\nLearning rate:%f Momentum: %f Max epoch: %d Hidden layers: (%s)",
							learningRate, momentum, maxEpoch, hiddenLayer));
					long time = System.currentTimeMillis();

					MultilayerPerceptron mlp = trainMLP(dataSet.trainingSet, learningRate, momentum, maxEpoch,
							hiddenLayer);
					time = System.currentTimeMillis() - time;

					logger.info(String.format("Training MLP took %d ms.", time));
					time = System.currentTimeMillis();
					Evaluation eval = evaluateMLP(mlp, dataSet.testSet);

					time = System.currentTimeMillis() - time;

					logger.info(String.format("Evaluation of MLP took %d ms.", time));

					logger.info(eval.toSummaryString());
					logger.info(eval.toClassDetailsString());
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	/**
	 * Reads data in arff format and returns instances.
	 * 
	 * @param path
	 *            of the arff file
	 * @return Instances object.
	 * @throws IOException
	 */
	public static Instances readData(String path) throws IOException {
		BufferedReader breader = null;
		breader = new BufferedReader(new FileReader(path));
		Instances data = new Instances(breader);
		data.setClassIndex(data.numAttributes() - 1);
		System.out.println(data.instance(0));
	
		return data;
	}

	/**
	 * Splits data into train and tests. Randomizes the instances if randomize is
	 * true.
	 * 
	 * @param instances
	 *            data to split
	 * @param ratio
	 *            of train/data size
	 * @param randomize
	 * @return DataSet object including training and test set.
	 */
	public static DataSet splitData(Instances instances, float ratio, boolean randomize) {
		if (randomize) {
			Random rand = new Random();
			instances.randomize(rand);
		}
		DataSet ret = new DataSet();
		
		int trainSize = (int) Math.round(instances.numInstances() * ratio);
		int testSize = instances.numInstances() - trainSize;
		ret.trainingSet = new Instances(instances, 0, trainSize);
		ret.testSet = new Instances(instances, trainSize, testSize);
		return ret;
	}

	/**
	 * According to the parameters generates a MLP and trains it with given train
	 * set.
	 * 
	 * @param trainSet
	 * @param learningRate
	 * @param momentum
	 * @param maxEpoch
	 * @param hiddenLayers
	 * @return trained MLP.
	 * @throws Exception
	 */
	public static MultilayerPerceptron trainMLP(Instances trainSet, float learningRate, float momentum, int maxEpoch,
			String hiddenLayers) throws Exception {
		MultilayerPerceptron trainedMLP = new MultilayerPerceptron();
		trainedMLP.setHiddenLayers(hiddenLayers);
		trainedMLP.setMomentum(momentum);
		trainedMLP.setLearningRate(learningRate);
		trainedMLP.setTrainingTime(maxEpoch);
		trainedMLP.buildClassifier(trainSet);
		return trainedMLP;
	}

	/**
	 * Evaluates given MLP with given test set.
	 * 
	 * @param mlp
	 * @param testSet
	 * @return evaluation results in Evaluation object.
	 * @throws Exception
	 */
	public static Evaluation evaluateMLP(Classifier mlp, Instances testSet) throws Exception {
		Evaluation evaluated = new Evaluation(testSet);
		evaluated.evaluateModel(mlp, testSet);
		return evaluated;
	}
}
