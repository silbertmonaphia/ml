package algorithms;

import algorithms.co_training.CoTrainingData;
import classificationResult.ClassificationResult;

/**
 * Abstract class representing an algorithm that can be run. This class should be inherited when adding a new algorithm. 
 */
public abstract class Algorithm {
	/**
	 * The Data to perform the experiment on. Set by call to {@link #setData} method at the beginning of {@link #run} method
	 */
	protected CoTrainingData data;
	/**
	 * Current fold of fold-cross validation
	 */
	protected int currentFold;
	/**
	 * Current feature split
	 */
	protected int currentSplit; 

	/**
	 * Running time of the algorithm (duration of running the <code>run</code> method)
	 */
	protected long runningTime;
	
	
	/**
	 * Returns the current state of data (labeled, unlabeled and test data)
	 * @return data for semi-supervised experiment (labeled, unlabeled and test data)
	 */
	public CoTrainingData getData() {
		return data;
	}
	
	/**
	 * Set the data to run experiment on. In this method, everything should be restarted to run the next experiment
	 * @param data separated labeled, unlabeled and test data
	 * @param fold current fold of the n-fold-cross validation  
	 * @param splitNo the current feature split run for the fold (e.g. in RSSalg for each fold, m feature splits are created for co-training)
	 * @param recordClassifiers whether to record statistics about building and testing each of the classifiers (see {@link #getClassifiers()} and {@link #getClassifiersTestData})
	 */
	protected void setData(CoTrainingData data, int fold, int splitNo){
		this.data = data;
		this.currentFold = fold;
		this.currentSplit = splitNo;
	}
	
	/**
	 * Run the algorithm
	 * @param data separated labeled, unlabeled and test data
	 * @param fold current fold of the n-fold-cross validation  
	 * @param splitNo the current feature split run for the fold (e.g. in RSSalg for each fold, m feature splits are created for co-training)
	 * @param recordClassifiers whether to record statistics about building and testing each of the classifiers (see {@link #getClassifiers()} and {@link #getClassifiersTestData})
	 * @return Classification result. The performance of the algorithm  is evaluated on the test 
	 * 			set defined in {@link CoTrainingData} object
	 * @throws Exception if there was an error running an algorithm
	 */
	public ClassificationResult run(CoTrainingData data, int fold, int splitNo) throws Exception {		
		setData(data, fold, splitNo);		
		return null;
	}
	
	
	/**
	 * Returns the name of the algorithm
	 * @return name of the algorithm. 
	 */
	public abstract String getName();
	
	
	/**
	 * Returns the algorithm running time. Running time is the time needed for the execution of {@link #run} method.
	 * @return running  time of the algorithm
	 */
	public long getRunningTimeMillis(){
		return runningTime;
	}
}
