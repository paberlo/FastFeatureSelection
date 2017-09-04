package test;

import static org.junit.Assert.*;

import java.io.FileReader;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Before;
import org.junit.Test;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.IWSS;
import weka.core.Instances;

/*Maven requires tests classes to contain sufix 'Test'*/
public class IWSSTest {
	Instances data;
	IWSS iwss;
	ASEvaluation evaluator;
	@Before
	public void setUp() throws Exception {
				
		data=new Instances(new FileReader("datasets/colon.arff"));
		data.setClassIndex(data.numAttributes()-1);
		iwss=new IWSS();
		evaluator=new weka.attributeSelection.WrapperSubsetEval();
	}

	@Test (expected = Exception.class)
	public void testexcep1() throws Exception {
		iwss.search(evaluator, null);
	}
	
	@Test (expected = Exception.class)
	public void testexcep2() throws Exception {
		iwss.search(null, data);
	}
	
	@Test
	public void testSearch() throws Exception{
		int[] atts=iwss.search(evaluator,data);
	
		assertTrue("attributes selected",atts.length>0);
	}
	

}
