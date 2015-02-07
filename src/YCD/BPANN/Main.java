package YCD.BPANN;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import LvShun.BP.BP2;

public class Main {

	
	public static void main(String[] args){
		long begin = System.currentTimeMillis();
		
		testBP();
//		testBP2();

		long delay = System.currentTimeMillis() - begin;
		System.out.println("耗时：" + delay + "ms");
		System.out.println("end");
	}
	
	public static void testBP(){
		System.out.println("begin");
		double[][] input = new double[3900][64];
		double[][] target = new double[3900][10];
		int[] label = new int[3900];
		readFile("digitstra.txt", input, target, label);
		BP bp = new BP(64, 118, 10, 1.7, 1.0);
		
		for (int k = 0; k < 2; k++) {
			for (int i = 0; i < input.length; i++) {
				bp.setEta(1.0 - 0.1 * k / 50);
				bp.train(input[i], target[i]);
			}
		}
		
		//-----------test---------
		double[][] testIn = new double[1500][64];
		double[][] testTar = new double[1500][10];
		int[] testLabel = new int[1500];
		readFile("digitstest.txt", testIn, testTar, testLabel);
		
		int correct = 0;
		int total = 1500;
		for (int i = 0; i < total; i++) {
			double[] result = bp.test(testIn[i]);
			int id = 0;
			for (int j = 0; j < result.length; j++) {
				if(result[id] < result[j]){
					id = j;
				}
			}
			if(id == testLabel[i]){
				correct ++;
			}
		}
		System.out.println("correct=" + correct + " rate=" + ((double)correct / total));
	}
	
	public static void testBP2(){
		System.out.println("begin");
		double[][] input = new double[3900][64];
		double[][] target = new double[3900][10];
		int[] label = new int[3900];
		readFile("digitstra.txt", input, target, label);
		BP2 bp2 = new BP2(64, 118, 10);
		
		for (int k = 0; k < 2; k++) {
			for (int i = 0; i < input.length; i++) {
				bp2.train(input[i], target[i], 1.7);
			}
		}
		
		double[][] testIn = new double[1500][64];
		double[][] testTar = new double[1500][10];
		int[] testLabel = new int[1500];
		readFile("digitstest.txt", testIn, testTar, testLabel);
		
		int correct = 0;
		int total = 1500;
		for (int i = 0; i < total; i++) {
			double[] result = bp2.test(testIn[i]);
			int id = 0;
			for (int j = 0; j < result.length; j++) {
				if(result[id] < result[j]){
					id = j;
				}
			}
			if(id == testLabel[i]){
				correct ++;
			}
		}
		System.out.println("correct=" + correct + " rate=" + ((double)correct / total));
	}
	
	public static void readFile(String filepath, double[][] input, double[][] target, int[] label){
		try {
			int row = 0;
			BufferedReader reader = new BufferedReader(new FileReader(new File(filepath)));
			String buff = null;
			while((buff = reader.readLine()) != null){
				String[] arr = buff.split(",");
				double max = Double.MIN_VALUE;
				double min = Double.MAX_VALUE;
				for (int i = 0; i < 64; i++) {
					double val = Double.parseDouble(arr[i]);
					input[row][i] = val;
					if(val > max){
						max = val;
					}
					if(val < min){
						min = val;
					}
				}
				
				//formalize input value
				if (max != min) {
					for (int i = 0; i < 64; i++) {
						input[row][i] = (input[row][i] - min) / (max - min);
					}
				} else if (max != 0) {
					for (int i = 0; i < 64; i++) {
						input[row][i] = (input[row][i]) / (max);
					}
				}
				
				//encoding output matrix
				for (int i = 0; i < 10; i++) {
					if(Integer.parseInt(arr[64]) == i){
						target[row][i] = 1;
					}else{
						target[row][i] = 0;
					}
				}
				
				label[row] = Integer.parseInt(arr[64]);
				
				row ++;
				if(row >= input.length){
					break;
				}
			}
			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
