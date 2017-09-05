

import static org.junit.Assert.*;

import java.io.FileReader;

import org.junit.Before;
import org.junit.Test;

import weka.attributeSelection.IGgaussian;
import weka.core.Instances;

/*Maven requires tests classes to contain sufix 'Test'*/
public class IGTest {
	Instances data1,data2;
	IGgaussian ig;
	
	@Before
	public void setUp() throws Exception {
		data1=new Instances(new FileReader("datasets/colon.arff"));
		data2=new Instances(new FileReader("datasets/adult.arff"));
		data1.setClassIndex(data1.numAttributes()-1);
		data2.setClassIndex(data2.numAttributes()-1);
		ig=new IGgaussian();
	}

	@Test
	public void testLearning() {
		try {
			ig.buildEvaluator(data1);
		} catch (Exception e) {
			fail("Exception learning gaussian with data1.");
		}	
		
		try {
			ig.buildEvaluator(data2);
			} catch (Exception e) {
				fail("Exception learning gaussian with data2.");
		}
		
	}

}
