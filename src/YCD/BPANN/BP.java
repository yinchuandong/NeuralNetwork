package YCD.BPANN;

import java.util.Arrays;
import java.util.Random;

public class BP {

	private double[] in;
	private double[] hide;
	private double[] out;
	private double[] target;
	
	private double[][] inHideW;
	private double[][] hideOutW;
	
	private double[] hideErr;
	private double[] outErr;
	
	private double outErrSum;
	private double hideErrSum;
	
	private double eta;
	/**
	 * 动量
	 */
	private double momentum = 1.0;
	/**
	 * 阀值
	 */
	private double b = 1.0;
	
	private Random random;
	
	public BP(int inSize, int hideSize, int outSize, double eta, double momentum){
		this.in = new double[inSize + 1];
		this.hide = new double[hideSize + 1];
		this.out = new double[outSize + 1];
		this.target = new double[outSize + 1];
		
		this.hideErr = new double[hideSize + 1];
		this.outErr = new double[outSize + 1];
		
		//increasing row size is to add W0*X0 as threshold
		this.inHideW = new double[hideSize + 1][inSize + 1];
		this.hideOutW = new double[outSize + 1][hideSize + 1];
		
		this.eta = eta;
		this.momentum = momentum;
		
		randomInit(inHideW);
		randomInit(hideOutW);
	}
	
	private void randomInit(double[][] weight){
		for (int j = 0; j < weight.length; j++) {
			for (int i = 1; i < weight[j].length; i++) {
				double real = random.nextDouble();
				weight[j][i] = real > 0.5 ? real : - real;
			}
		}
	}
	
	public void train(double[] trainData, double[] target){
		loadTrain(trainData);
		loadTarget(target);
		forward();
		backProprogate();
		adjustWeight();
	}
	
	public double[] predict(double[] inData){
		if(inData.length != in.length - 1){
			throw new IllegalArgumentException("In Size Do Not Match.");
		}
		System.arraycopy(inData, 0, in, 1, inData.length);
		forward();
		return getNetworkOutput();
	}
	
	/**
	 * return the value of output layer
	 */
	private double[] getNetworkOutput() {
		int len = out.length;
		double[] temp = new double[len - 1];
		for (int i = 1; i != len; i++)
			temp[i - 1] = out[i];
		return temp;
	}
	
	private void loadTrain(double[] trainData){
		if(trainData.length != in.length - 1){
			throw new IllegalArgumentException("Train Size Do Not Match.");
		}
		System.arraycopy(trainData, 0, in, 1, trainData.length);
	}
	
	private void loadTarget(double[] t){
		if(t.length != target.length - 1){
			throw new IllegalArgumentException("Train Size Do Not Match.");
		}
		System.arraycopy(t, 0, target, 1, t.length);
	}
	
	private void forward(){
		forward(in, hide, inHideW);
		forward(hide, out, hideOutW);
	}
	
	private void backProprogate(){
		calcOutErr();
		calcHideError();
	}
	
	private void adjustWeight(){
		adjustWeight(hide, outErr, hideOutW);
		adjustWeight(in, hideErr, inHideW);
	}
	
	private void forward(double[] in, double[] out, double[][] w){
		in[0] = 1.0;
		for (int j = 1; j < out.length; j++) {
			double sum = 0.0;
			for (int i = 0; i < in.length; i++) {
				sum += w[j][i] * in[i];
			}
			out[j] = sigmoid(sum);
		}
	}
	
	private void calcOutErr(){
		double sum = 0.0;
		for (int i = 1; i < outErr.length; i++) {
			double t = target[i];
			double o = out[i];
			double e = o * (1 - o) * (t - o);
			outErr[i] = e;
			sum += Math.abs(e);
		}
		this.outErrSum = sum;
	}
	
	private void calcHideError(){
		double sum = 0.0;
		for (int i = 1; i < hideErr.length; i++) {
			double o = hide[i];
			double tmpS = 0.0;
			for (int j = 0; j < outErr.length; j++) {
				tmpS += hideOutW[j][i + 1] * out[j];
			}
			double e = o * (1 - o) * tmpS;
			hideErr[i] = e;
			sum += e;
		}
		this.hideErrSum = sum;
	}
	
	private void adjustWeight(double[] in, double[] delta, double[][] w){
		for (int j = 1; j < delta.length; j++) {
			for (int i = 0; i < in.length; i++) {
				double deltaW = eta * delta[j] * in[i] + momentum * w[j][i];
				w[j][i] += deltaW;
			}
		}
	}
	
	private double sigmoid(double x){
		return 1.0 / (1.0 + Math.exp( - x ));
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}
