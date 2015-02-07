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
	
	/**
	 * save the old delta weight of in-hide layer
	 */
	private double[][] inHideOldW;
	/**
	 * save the old delta weight of hide-out layer
	 */
	private double[][] hideOutOldW;
	
	private double[] hideDelta;
	private double[] outDelta;
	
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
		
		this.hideDelta = new double[hideSize + 1];
		this.outDelta = new double[outSize + 1];
		
		//increasing row size is to add W0*X0 as threshold
		this.inHideW = new double[hideSize + 1][inSize + 1];
		this.hideOutW = new double[outSize + 1][hideSize + 1];
		this.inHideOldW = new double[hideSize + 1][inSize + 1];
		this.hideOutOldW = new double[outSize + 1][hideSize + 1];
		
		this.eta = eta;
		this.momentum = momentum;
		
		this.random = new Random();
		randomInit(inHideW);
		randomInit(hideOutW);
	}
	
	private void randomInit(double[][] weight){
		for (int j = 0; j < weight.length; j++) {
			for (int i = 0; i < weight[j].length; i++) {
				double real = random.nextDouble();
				weight[j][i] = real > 0.5 ? real : - real;
			}
		}
	}
	
	public void setEta(double eta){
		this.eta = eta;
	}
	
	public void train(double[] trainData, double[] target){
		loadTrain(trainData);
		loadTarget(target);
		forward();
		backProprogate();
		adjustWeight();
	}
	
	public double[] test(double[] inData){
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
		adjustWeight(hide, outDelta, hideOutW, hideOutOldW);
		adjustWeight(in, hideDelta, inHideW, inHideOldW);
	}
	
	private void forward(double[] in, double[] out, double[][] w){
		in[0] = 1.0;
		for (int j = 1; j < out.length; j++) {
			double sum = 0.0;
			for (int i = 0; i < in.length; i++) {
				sum += w[j][i] * in[i];
			}
			double t = sigmoid(sum);
			out[j] = t;
		}
	}
	
	private void calcOutErr(){
		double sum = 0.0;
		for (int j = 1; j < outDelta.length; j++) {
			double t = target[j];
			double o = out[j];
//			double e = o * (1.0 - o) * (t - o);
			double e = Math.exp(-o) / ((1 + Math.exp(-o)) * (1 + Math.exp(-o))) * (t - o);
			outDelta[j] = e;
			sum += Math.abs(e);
		}
		this.outErrSum = sum;
	}
	
	private void calcHideError(){
		double sum = 0.0;
		for (int j = 1; j < hideDelta.length; j++) {
			double o = hide[j];
			double tmpS = 0.0;
			for (int k = 1; k < outDelta.length; k++) {
				tmpS += hideOutW[k][j] * outDelta[k];
			}
//			double e = o * (1.0 - o) * tmpS;
			double e = Math.exp(-o) / ((1 + Math.exp(-o)) * (1 + Math.exp(-o))) * tmpS;
			hideDelta[j] = e;
			sum += e;
		}
		this.hideErrSum = sum;
	}
	
	private void adjustWeight(double[] in, double[] delta, double[][] w, double[][] oldW){
		for (int j = 1; j < delta.length; j++) {
			for (int i = 0; i < in.length; i++) {
				double deltaW = eta * delta[j] * in[i] + momentum * oldW[j][i];
				w[j][i] += deltaW;
				oldW[j][i] = deltaW;
			}
		}
	}
	
	private double sigmoid(double x){
		return 1d / (1d + Math.exp( - x ));
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}
