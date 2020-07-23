package proccess;

import util.Util;
import weka.core.Instances;

public class CombineSample {
	public static void main(String[] args)throws Exception {
		Instances data=Util.getInstances("../tmp/8file_train.csv");
		Instances sa=Util.getInstances("../tmp/sample400.csv");
		data=Util.addAll(data,sa);
		Util.saveIns(data, "../tmp/8file_train_sample.csv");
	}
}
