package experimentSetting;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import util.PropertiesReader;
import algorithms.Algorithm;
//import algorithms.SupervisedAlgorithm_All;
//import algorithms.SupervisedAlgorithm_L;
//import algorithms.RSSalg.MajorityVote;
//import algorithms.RSSalg.RSSalg;
//import algorithms.RSSalg.GA.CandidateEvaluatorIF;
//import algorithms.RSSalg.voter.VoterIF;
import classificationResult.measures.MeasureIF;
//import featureSplit.SplitterIF;

/**
 * Class encapsulating all experiment settings: algorithm which is run (e.g. co-training or RSSalg), 
 * feature splitting algorithm, etc.
 * Singleton object   
 */
public class ExperimentSettings {
	private static ExperimentSettings instance = null;
	protected Algorithm algorithm = null; // Algorithm that will be run (currently supports RSSalg and co-training)
	
	protected boolean balancedSplit = true; // for now, only ballanced split is supported
	
	// Number of different splits used with co-training (parameter m in RSSalg) or number of Random Splits. This parameter is specified
	// only if RSSalg or Co-training wRandom is run (otherwise it is ignored) 
	protected int noSplits = 1;
	// The measures to be calculated in the experiment
	protected List<MeasureIF> measures = new ArrayList<MeasureIF>();
	protected boolean writeEnlargedCoTrainingSet = false;
	
	public static ExperimentSettings getInstance() {
		if(instance == null) {
			instance = new ExperimentSettings();
	    }
	    return instance;
	}          
	
	public boolean isInitiated(){
		return algorithm != null;
	}
	
	public void clear(){
		algorithm = null;
		balancedSplit = true;
		noSplits = 1;
		measures.clear();
		writeEnlargedCoTrainingSet = false;
		System.out.println("Experiment settings cleared.");
	}

	public Algorithm getAlgorithm() {
		return algorithm;
	}
	private void setAlgorithm(Algorithm algorithm) {
		this.algorithm = algorithm;
	}

	public boolean isBalancedSplit() {
		return balancedSplit;
	}
	public int getNoSplits() {
		return noSplits;
	}
	private void setNoSplits(int noSplits) throws Exception {
		if(noSplits < 1)
			throw new Exception("There must be at least 1 feature split for co-training. Trying to set " + noSplits + ")");
		this.noSplits = noSplits;
	}
	public List<MeasureIF> getMeasures() {
		return measures;
	}
	private void addMeasure(MeasureIF measure) {
		this.measures.add(measure);
	}
	public boolean isWriteEnlargedCoTrainingSet() {
		return writeEnlargedCoTrainingSet;
	}
	private void setWriteEnlargedCoTrainingSet(boolean writeEnlargedCoTrainingSet) {
		this.writeEnlargedCoTrainingSet = writeEnlargedCoTrainingSet;
	}

	public void readProperties(String propertiesFile) throws Exception{
		Properties properties = null;
		try {
			properties = new Properties();
			properties.load(new FileInputStream(propertiesFile));
		}catch (FileNotFoundException e) {
			throw new Exception("ERROR: error reading properties file: file " + propertiesFile + "does not exist", e);
		}
		clear();
		System.out.println("Reading the experiment settings from file: " + propertiesFile);
		
		
		setAlgorithm((Algorithm) PropertiesReader.readObjectParam(properties, "algorithm"));
		
		List<String> measureClassNames = PropertiesReader.readStringListParam(properties, "measures");
		List<String> measuresForClasses = PropertiesReader.readStringListParam(properties, "measuresForClass");
		if(measureClassNames.size() != measuresForClasses.size()){
			throw new Exception("ERROR: a class needs to be specified for each measure (\"avg\" if the measure does not depend on a class).");
		}
		for(int i=0; i<measureClassNames.size(); i++){
			MeasureIF measure = (MeasureIF) PropertiesReader.getObject(measureClassNames.get(i));
			if(!measure.dependsOnClass()){
				addMeasure(measure);
			}else{
				measure.setClassName(measuresForClasses.get(i));
				addMeasure(measure);
			}
		}
		
		setWriteEnlargedCoTrainingSet(PropertiesReader.readBooleanParam(properties, "writeEnlargedTrainingSet"));
		
		
		String resultFolder = "";
		try{
			resultFolder = DatasetSettings.getInstance().getResultFolder();
			PrintStream writer = new PrintStream(new FileOutputStream(resultFolder + "/Experiment.txt", true)); 
			printSettings(writer);
			writer.close();
		}catch(Exception e){
			System.out.println("WARNING: could not write the data settings for the experiment in file " + resultFolder + "/Experiment.txt");
		}
	} 
	
	/**
	 * Print the experiment setting
	 * @param out : PrintStream for writing   
	 */
	public void printSettings(PrintStream out){	
		out.println("EXPERIMENT SETTINGS:");
		out.println("\tRunning " + algorithm.getClass().getName() + " algorithm");
		out.println("\tRunning " + noSplits + " splits with co-training");	
	}
}
