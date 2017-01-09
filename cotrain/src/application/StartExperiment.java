package application;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.List;
import resultsToXML.ExperimentResults;
import resultsToXML.Experiments;
import resultsToXML.Measure;
import setExperiment.CrossValidationSeparator;
import algorithms.Algorithm;
import algorithms.co_training.CoTrainingData;
import classificationResult.ClassificationResult;
import classificationResult.measures.MeasureIF;
import experimentSetting.CVSettings;
import experimentSetting.CoTrainingSettings;
import experimentSetting.DatasetSettings;
import experimentSetting.ExperimentSettings;

/**
 * The main class that starts the experiment
 */
public class StartExperiment {

	/**
	 * Reads experiment properties and starts the experiment
	 * 
	 * @param pathToPropertiesFolder
	 *            the path to the folder that contains experiment properties
	 *            (cv.properties, data.properties, co-training.properties,
	 *            GA.properties)
	 * @param experimentSettingsFile
	 *            experiment properties file name
	 * @throws Exception
	 *             if there is an error reading one of the properties files
	 */
	public static void setExperiment(String pathToPropertiesFolder, String experimentSettingsFile) throws Exception {
		String filePath = pathToPropertiesFolder + "/";

		try {
			DatasetSettings.getInstance().readProperties(filePath + "data.properties");
		} catch (Exception e) {
			throw new Exception("ERROR: Cannot read data properties", e);
		}

		try {
			CVSettings.getInstance().readProperties(filePath + "cv.properties");
		} catch (Exception e) {
			throw new Exception("ERROR: Cannot read cv properties", e);
		}

		try {
			CoTrainingSettings.getInstance().readProperties(filePath + "co-training.properties");
		} catch (Exception e) {
			throw new Exception("ERROR: Cannot read co-training properties", e);
		}

		try {
			ExperimentSettings.getInstance().readProperties(filePath + experimentSettingsFile);
		} catch (Exception e) {
			throw new Exception("ERROR: Cannot read experiment properties", e);
		}

	}

	/**
	 * Sets the cross-validation experiment. Depending on the properties
	 * settings, reads a preset cross-validation experiment or prepares the data
	 * for a new cross-validation experiment
	 * 
	 * @throws Exception
	 *             if:
	 *             <ul>
	 *             <li>the ARFF file is missing
	 *             <li>class attribute is missing (there is no attribute in the
	 *             dataset that matches the name of the class attribute given in
	 *             the data properties)
	 *             <li>adding an id attribute failed
	 *             </ul>
	 */
	private void setCrossValidationExperiment() throws Exception {
		if (DatasetSettings.getInstance().isLoadPresetExperiment()) // do not
																	// create
																	// folds
			return;

		System.out.println("Creating folds...");
		CrossValidationSeparator cvSeparator = new CrossValidationSeparator();
		CoTrainingData[] data = cvSeparator.prepareCrossValidationExperiment();

		String resultFolder = DatasetSettings.getInstance().getResultFolder();
		for (int i = 0; i < data.length; i++) {
			try {
				data[i].saveData(resultFolder + "/fold_" + i + "/");
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		System.out.println("Folds created.");
	}

	/**
	 * Returns the number of folds in the experiment
	 * 
	 * @return number of folds
	 */
	private int getNoFoldsInResultFolder() {
		String path = DatasetSettings.getInstance().getResultFolder();
		int count = 0;
		while (true) {
			File subfolder = new File(path + "/fold_" + count);
			if (subfolder.exists())
				count++;
			else
				break;
		}
		return count;
	}

	/**
	 * Loads one fold data for n-fold-cross validation
	 * 
	 * @param fold
	 *            fold number
	 * @return fold data
	 * @throws Exception
	 *             if there was an error loading the fold
	 */
	private CoTrainingData loadFold(int fold) throws Exception {
		String path = DatasetSettings.getInstance().getResultFolder();
		File subfolder = new File(path + "/fold_" + fold);
		System.out.println("Reading " + subfolder.getPath());
		return new CoTrainingData(subfolder.getPath(), DatasetSettings.getInstance().getNoViews(), true);
	}

	/**
	 * Runs the cross-validation experiment
	 * 
	 * @throws Exception
	 *             if:
	 *             <ul>
	 *             <li>There was an error reading/writing the results from/to
	 *             Results.xml file
	 *             <li>A feature splitting algorithm is required, but not
	 *             specified
	 *             <li>There was an error creating the feature split
	 *             <li>There was an error running the algorithm
	 *             </ul>
	 */
	private void runCrossvalidation() throws Exception {
		ExperimentResults results = new ExperimentResults();
		results.fromXML(DatasetSettings.getInstance().getResultFolder() + "/Results.xml");

		List<MeasureIF> measures = ExperimentSettings.getInstance().getMeasures();
		Algorithm algorithm = ExperimentSettings.getInstance().getAlgorithm();
		int noSplits = ExperimentSettings.getInstance().getNoSplits();
		if (algorithm.getName().contains("_of_Co-training_classifiers_on_test_set"))
			noSplits = 1;

		DecimalFormat df = new DecimalFormat("###.#");
		System.out.println();
		System.out.println("Starting cross-validation for " + algorithm.getName() + " experiment...");
		System.out.println();
		int noFolds = getNoFoldsInResultFolder();

		ClassificationResult microAveragedResult = new ClassificationResult(false);
		double[][] macroAveragedResult = new double[measures.size()][noFolds];

		for (int i = 0; i < noFolds; i++) {
			System.out.println();
			System.out.println("Starting Fold " + i);
			CoTrainingData data = loadFold(i);

			for (int split = 0; split < noSplits; split++) {
				CoTrainingData tmpData = new CoTrainingData(data);

				ClassificationResult result = algorithm.run(tmpData, i, split);
				microAveragedResult.updateResults(result);

				System.out.println("Split " + split + ":");
				for (int measInd = 0; measInd < measures.size(); measInd++) {
					String measureName = measures.get(measInd).getName();
					Double measureValue = measures.get(measInd).getMeasure(result);
					System.out.println("\t" + measureName + ": " + df.format(measureValue));
					macroAveragedResult[measInd][i] = measureValue;
				}
				result = null;
			}

		}

		System.out.println();
		System.out.println("Experiment finished.");
		System.out.println();

		Experiments experiments = results.findExperimentsByProperties();
		String expName = algorithm.getName();
		Experiments.Experiment newEx = experiments.findExperiment(expName);

		for (int measureInd = 0; measureInd < measures.size(); measureInd++) {
			System.out.println(measures.get(measureInd).getName());
			System.out.println(
					"\tmicro averaged: " + df.format(measures.get(measureInd).getMeasure(microAveragedResult)));

			double avgMesure = 0;
			for (int foldInd = 0; foldInd < macroAveragedResult[measureInd].length; foldInd++) {
				avgMesure += macroAveragedResult[measureInd][foldInd];
			}
			avgMesure /= macroAveragedResult[measureInd].length;

			double variance = 0;
			for (int foldInd = 0; foldInd < macroAveragedResult[measureInd].length; foldInd++) {
				variance += (macroAveragedResult[measureInd][foldInd] - avgMesure)
						* (macroAveragedResult[measureInd][foldInd] - avgMesure);
			}
			variance /= (macroAveragedResult[measureInd].length - 1);

			System.out.println("\tmacro averaged: " + df.format(avgMesure) + " +/- " + df.format(Math.sqrt(variance)));

			Measure measure = newEx.findMeasure(measures.get(measureInd).getName());
			measure.setMicroAveraged(measures.get(measureInd).getMeasure(microAveragedResult));
			measure.setMacroAveraged(avgMesure);
			measure.setStdDev(Math.sqrt(variance));
		}
		results.toXML(new FileOutputStream(new File(DatasetSettings.getInstance().getResultFolder() + "/Results.xml")));
		System.out.println();
		System.out.println(results);

	}

	public void run() throws Exception {
		setCrossValidationExperiment();
		runCrossvalidation();
	}

	public static void main(String[] args) {
		// 多种不同的实验设置，第二个参数用来指定不同的实验
		// setExperiment("./data/News2x2/experiment","experiment_L.properties");
		// setExperiment("./data/News2x2/experiment","experiment_All.properties");
		// setExperiment("./data/News2x2/experiment","experiment_RSSalg.properties");
		// setExperiment("./data/News2x2/experiment","experiment_RSSalg_best.properties");
		// setExperiment("./data/News2x2/experiment","experiment_Co-training_Random.properties");
		// setExperiment("./data/News2x2/experiment","experiment_MV.properties");
		// setExperiment("./data/News2x2/experiment","experiment_Co-training_Natural.properties");

		try {
			setExperiment("D:/EclipseJ2EEWorkspace/CoTraining20160923/data/News2x2/experiment",
					"experiment_Co-training_Natural.properties");
			StartExperiment experimentStarter = new StartExperiment();
			experimentStarter.run();
		} catch (Exception e) {
			Throwable cause = e;
			while (cause.getCause() != null) {
				cause = cause.getCause();
			}
			System.out.println(cause.getMessage());
		}
	}

}
