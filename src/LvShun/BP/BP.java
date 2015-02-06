package LvShun.BP;

import java.util.Random;

/**
 * BPNN.一个两层的反馈神经网络 Back Propagation NerseNet
 * 2014.12.29
 *
 * @author LvShun
 *
 */
public class BP {
	/**
	 * input vector.
	 */
	private final double[] input;
	/**
	 * hidden layer.
	 */
	private final double[] hidden;
	/**
	 * output layer.
	 */
	private final double[] output;
	/**
	 * target.
	 */
	private final double[] target;

	/**
	 * delta vector of the hidden layer .
	 */
	private final double[] hidDelta;
	/**
	 * output layer of the output layer.
	 */
	private final double[] optDelta;

	/**
	 * learning rate/ bias/ threshold 偏置或阈值(反正叫啥的都有)
	 */
	private double eta;
	/**
	 * weight matrix from input layer to hidden layer.
	 */
	private final double[][] iptHidWeights;
	/**
	 * weight matrix from hidden layer to output layer.
	 */
	private final double[][] hidOptWeights;

	/**
	 * previous weight update.
	 */
	private final double[][] iptHidPrevUptWeights;
	/**
	 * previous weight update.
	 */
	private final double[][] hidOptPrevUptWeights;

	public double optErrSum = 0d;

	public double hidErrSum = 0d;

	private final Random random;

	/**
	 * 构造函数 这里把阈值直接加入的结果 所以数组大小要加1
	 *
	 */
	public BP(int inputSize, int hiddenSize, int outputSize, double eta) {

		input = new double[inputSize + 1];
		hidden = new double[hiddenSize + 1];
		output = new double[outputSize + 1];
		target = new double[outputSize + 1];

		hidDelta = new double[hiddenSize + 1];
		optDelta = new double[outputSize + 1];

		iptHidWeights = new double[inputSize + 1][hiddenSize + 1];
		hidOptWeights = new double[hiddenSize + 1][outputSize + 1];

		random = new Random();
		randomizeWeights(iptHidWeights);
		randomizeWeights(hidOptWeights);

		iptHidPrevUptWeights = new double[inputSize + 1][hiddenSize + 1];
		hidOptPrevUptWeights = new double[hiddenSize + 1][outputSize + 1];

		this.eta = eta;
	}

	/**
	 * 随机赋予权值【-1,1】
	 *
	 */
	private void randomizeWeights(double[][] matrix) {
		for (int i = 0, len = matrix.length; i != len; i++)
			for (int j = 0, len2 = matrix[i].length; j != len2; j++) {
				double real = random.nextDouble();
				matrix[i][j] = real > 0.5 ? real : - real;
			}
	}

	/**
	 * 相当于一个默认构造函数  eta = 0.25 and momentum = 0.3.
	 *
	 *
	 */
	public BP(int inputSize, int hiddenSize, int outputSize) {
		this(inputSize, hiddenSize, outputSize, 1.7);
	}

	/**
	 * 训练模型 输入和输出的都是一维的数组.
	 *
	 */
	public void train(double[] trainData, double[] target, double thea) {
		this.eta = thea;
		loadInput(trainData);
		loadTarget(target);
		forward();
		calculateDelta();
		adjustWeight();
	}

	/**
	 * 用于测试 BPNN 输入一个一位数组 按照网络输出一个数组 与外部的结果相比.
	 *
	 */
	public double[] test(double[] inData) {
		if (inData.length != input.length - 1) {
			throw new IllegalArgumentException("Size Do Not Match.");
		}
		System.arraycopy(inData, 0, input, 1, inData.length);
		forward();
		return getNetworkOutput();
	}

	/**
	 * 返回输出层的值.
	 *
	 */
	private double[] getNetworkOutput() {
		int len = output.length;
		double[] temp = new double[len - 1];
		for (int i = 1; i != len; i++)
			temp[i - 1] = output[i];
		return temp;
	}

	/**
	 * 加载目标输出值.
	 *
	 */
	private void loadTarget(double[] arg) {
		if (arg.length != target.length - 1) {
			throw new IllegalArgumentException("Size Do Not Match.");
		}
		System.arraycopy(arg, 0, target, 1, arg.length);
	}

	/**
	 * 加载训练数据.
	 *
	 */
	private void loadInput(double[] inData) {
		if (inData.length != input.length - 1) {
			throw new IllegalArgumentException("Size Do Not Match.");
		}
		System.arraycopy(inData, 0, input, 1, inData.length);
	}


	/**
	 * 两层前馈.
	 */
	private void forward() {
		forward(input, hidden, iptHidWeights,hidDelta);
		forward(hidden, output, hidOptWeights,optDelta);
	}

	/**
	 * 前馈，由输入和权值得到输出（两层通用）.
	 */
	private void forward(double[] layer0, double[] layer1, double[][] weight, double[] d) {
		//阈值.
		layer0[0] = 1.0;
		for (int j = 1, len = layer1.length; j != len; ++j) {
			double sum = 0;
			for (int i = 0, len2 = layer0.length; i != len2; ++i)
				sum += weight[i][j] * layer0[i];
			layer1[j] = sigmoid(sum);
		}
	}

	/**
	 * 误差反馈.
	 */
	private void calculateDelta() {
		outputErr();
		hiddenErr();
	}

	/**
	 * 计算输出层误差数组（一维）.
	 */
	private void outputErr() {
		double errSum = 0;
		for (int idx = 1, len = optDelta.length; idx != len; ++idx) {
			double o = output[idx];
			optDelta[idx] = Math.exp(-o) / ((1 + Math.exp(-o)) * (1 + Math.exp(-o))) * (target[idx] - o);
			errSum += Math.abs(optDelta[idx]);
		}
		optErrSum = errSum;
	}

	/**
	 * 计算隐层的误差数组（一维）.
	 *
	 */
	private void hiddenErr() {
		double errSum = 0;
		for (int j = 1, len = hidDelta.length; j != len; ++j) {
			double o = hidden[j];
			double sum = 0;
			for (int k = 1, len2 = optDelta.length; k != len2; ++k)
				sum += hidOptWeights[j][k] * optDelta[k];
			hidDelta[j] = Math.exp(-o) / ((1 + Math.exp(-o)) * (1 + Math.exp(-o))) * sum;
			errSum += Math.abs(hidDelta[j]);
		}
		hidErrSum = errSum;
	}


	/**
	 * 调整权值（两层通用）.
	 */
	private void adjustWeight(double[] delta, double[] layer,
							  double[][] weight, double[][] prevWeight) {

		layer[0] = 1;
		for (int i = 1, len = delta.length; i != len; ++i) {
			for (int j = 0, len2 = layer.length; j != len2; ++j) {
				double newVal = prevWeight[j][i] + eta * delta[i] * layer[j];
				weight[j][i] += newVal; //eta * delta[i] * layer[j];
				prevWeight[j][i] = newVal;
			}
		}
	}

	/**
	 * 反向调整权值.
	 */
	private void adjustWeight() {
		adjustWeight(optDelta, hidden, hidOptWeights, hidOptPrevUptWeights);
		adjustWeight(hidDelta, input, iptHidWeights, iptHidPrevUptWeights);
	}

	/**
	 * S形函数【0,1】
	 *
	 */
	private double sigmoid(double val) {
		return 1d / (1d + Math.exp(-val));
	}
}