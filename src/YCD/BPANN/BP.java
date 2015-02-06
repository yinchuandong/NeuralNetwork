package YCD.BPANN;

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
	
	public BP(int inSize, int hideSize, int outSize, double eta, double momentum, double b){
		this.in = new double[inSize];
		this.hide = new double[hideSize];
		this.out = new double[outSize];
		this.target = new double[outSize];
		
		this.hideErr = new double[hideSize];
		this.outErr = new double[outSize];
		
		//increasing row size is to add W0*X0 as threshold
		this.inHideW = new double[hideSize][inSize + 1];
		this.hideOutW = new double[outSize][hideSize + 1];
		
		this.eta = eta;
		this.momentum = momentum;
		this.b = b;
		
		randomInit(inHideW);
		randomInit(hideOutW);;
	}
	
	private void randomInit(double[][] weight){
		for (int i = 0; i < weight.length; i++) {
			for (int j = 0; j < weight[i].length; j++) {
				double real = random.nextDouble();
				weight[i][j] = real > 0.5 ? real : - real;
			}
		}
	}
	
	public void train(double[] in, double[] target){
		this.in = in;
		this.target = target;
		
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
		for (int j = 0; j < w.length; j++) {
			double sum = b * w[j][0];
			for (int i = 1; i < w[j].length; i++) {
				sum += w[j][i] * in[i];
			}
			out[j] = sigmoid(sum);
		}
	}
	
	private void calcOutErr(){
		double sum = 0.0;
		for (int i = 0; i < outErr.length; i++) {
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
		for (int i = 0; i < hideErr.length; i++) {
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
		for (int j = 0; j < delta.length; j++) {
			for (int i = 0; i < in.length; i++) {
				double xji = (i == 0 ? b : in[i]);
				double deltaW = eta * delta[j] * xji + momentum * w[j][i];
				w[j][i] += deltaW;
			}
		}
	}
	
	private double sigmoid(double x){
		return 1.0 / (1.0 + Math.exp( - x ));
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}
