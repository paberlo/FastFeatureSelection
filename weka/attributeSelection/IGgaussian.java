package weka.attributeSelection;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Capabilities.Capability;


import java.util.Enumeration;

/**
 * Computes IG between a numerical attribute X and a multinomial class C without
 * performing any discretization. see: "Supervised classification with
 * conditional Gaussian networks: Increasing the structure complexity from Naive
 * Bayes. Aritz Perez, Pedro Larrañaga, Iñaki Inza."
 * 
 * If X is numerical: InfoGain(X,C) = 1/2(log(variance(X)) - sum_c=1^r
 * P(c)log(variance(X|c)) if X is nominal: InfoGain(X,C) = H(X) - H(X|C)";
 * 
 * 
 * @author pablo.bermejo@uclm.es
 * 
 */

public class IGgaussian extends ASEvaluation implements AttributeEvaluator,
		OptionHandler {

	/** for serialisation */
	static final long serialVersionUID = -174552112589257126L;

	/** The info gain for each attribute */
	private double[] m_InfoGains;

	/**
	 * Returns a string describing this attribute evaluator
	 * 
	 * @return a description of the evaluator suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {
		return "IGgaussian :\n\nEvaluates the worth of a numerical or nominal attribute "
				+ "by measuring the information gain with respect to a multinomial class,"
				+ "without performing any discretization.\n\n"
				+ " For numerical atts: InfoGain(X,C) = 1/2(log(variance(X)) - sum_c=1^r P(c)log(variance(X|c))"
				+ "\n\n For nominal atts: InfoGain(X,C) =  H(X) - H(X|C)";

	}

	@SuppressWarnings("rawtypes")
	@Override
	public Enumeration listOptions() {
		// no options
		return null;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		// no options

	}

	@Override
	public String[] getOptions() {
		// no options
		return new String[0];

	}

	/**
	 * Reset options to their default values
	 */
	protected void resetOptions() {
		// no options
	}

	/**
	 * Returns the capabilities of this evaluator.
	 * 
	 * @return the capabilities of this evaluator
	 * @see Capabilities
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		return result;
	}

	/**
	 * Initialises an information gain attribute evaluator. No discretization is
	 * performed.
	 * 
	 * For numerical atts: InfoGain(X,C) = 1/2(log(variance(X)) - sum_c=1^r
	 * P(c)log(variance(X|c))
	 * 
	 * For nominal atts: InfoGain(X,C) = H(X) - H(X|C)
	 * 
	 * 
	 * @param data
	 *            set of instances serving as training data
	 * @throws Exception
	 *             if the evaluator has not been generated successfully
	 */
	@Override
	public void buildEvaluator(Instances data) throws Exception {
		// can evaluator handle data?
		getCapabilities().testWithFail(data);

		int classIndex = data.classIndex();
		int numClasses = data.attribute(classIndex).numValues();
		int numInstances = data.numInstances();
		int[] classCounts = data.attributeStats(classIndex).nominalCounts;
		double counts[][][] = new double[data.numAttributes() - 1][numClasses][];

		m_InfoGains = new double[data.numAttributes()];
		InfoGainAttributeEval IGnominal = null;

		// get P(c)
		double probClass[] = new double[numClasses];
		for (int c = 0; c < numClasses; c++) {
			probClass[c] = ((double) classCounts[c]) / ((double) numInstances);
		}

		// init counts
		if (data.checkForAttributeType(Attribute.NUMERIC)) {
			for (int i = 0; i < data.numAttributes() - 1; i++) {
				for (int c = 0; c < numClasses; c++) {
					counts[i][c] = new double[classCounts[c]];

				}
			}
		}

		// get counts
		data.sort(data.classIndex());
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			// numeric attribute
			if (data.attribute(i).isNumeric()) {
				int pointer = 0;
				for (int c = 0; c < numClasses; c++) {
					for (int n = 0; n < classCounts[c]; n++) {
						counts[i][c][n] = data.instance(pointer).value(i);
						pointer++;
					}
				}

			}
			// handle nominal attributes
			if (IGnominal == null && data.attribute(i).isNominal()) {
				IGnominal = new InfoGainAttributeEval();
				IGnominal.buildEvaluator(data);
			}
		}

		// apply IG formula
		for (int k = 0; k < data.numAttributes(); k++) {
			if (k != classIndex) {
				if (data.attribute(k).isNumeric()) {
					double var = data.variance(k);
					double sum = 0;
					for (int c = 0; c < numClasses; c++) {
						sum += (probClass[c] * Math.log(weka.core.Utils
								.variance(counts[k][c])));
					}
					m_InfoGains[k] = (Math.log(var) - sum) / 2;
				} else {
					m_InfoGains[k] = IGnominal.evaluateAttribute(k);

				}

			}
		}

	}

	/**
	 * evaluates an individual attribute by measuring the amount of information
	 * gained about the class given the attribute.
	 * 
	 * @param attribute
	 *            the index of the attribute to be evaluated
	 * @return the info gain
	 * @throws Exception
	 *             if the attribute could not be evaluated
	 */
	public double evaluateAttribute(int attribute) throws Exception {

		return m_InfoGains[attribute];
	}

	/**
	 * Describe the attribute evaluator
	 * 
	 * @return a description of the attribute evaluator as a string
	 */
	public String toString() {
		StringBuffer text = new StringBuffer();

		if (m_InfoGains == null) {
			text.append("Information Gain attribute evaluator has not been built");
		} else {
			text.append("\tInformation Gain Ranking Filter");

		}
		text.append("\n");
		return text.toString();

	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 1 $");
	}

	// ============
	// Test method.
	// ============
	/**
	 * Main method for testing this class.
	 * 
	 * @param args
	 *            the options [-i,fileRoute]
	 */
	public static void main(String[] args) {

		
			runEvaluator(new IGgaussian(), args);
			

	}
}