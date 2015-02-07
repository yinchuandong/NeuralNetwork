package LvShun.BP;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import YCD.BPANN.BP;


/**
 * Test BPNN.
 *
 * @author LvShun
 */
public class Test {

    /**
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
//         for(int i=130;i>5;i--){
//
//         System.out.print(i+" ");

        int total = -1;
        double[][] input = new double[3900][64];
        double[][] output = new double[3900][10];
        BufferedReader reader;
        try {
            System.out.println("正在读取训练集...");
            FileReader file = new FileReader("digitstra.txt");
            System.out.println("读取完成，正在训练神经网络！");
            reader = new BufferedReader(file);
            String line = reader.readLine();
            while (line != null) {
                total++;
                //处理这一行读出的东西并存进两个数组，完成训练
                int con = 0;//逗号的个数
                int count = 0;//字符的个数
                int tt = 0;//取出的数字
                while (con < 64) {
                    char tmp = line.charAt(count);//取出的字符
                    if (tmp != ',') {
                        tt = 10 * tt + (tmp - '0');
                        count++;
                    } else if (con != 63) {
                        input[total][con++] = (double) tt;
                        tt = 0;
                        count++;
                    } else {
                        input[total][con++] = (double) tt;
                        double max = 0;
                        double min = 16;
                        for (int k = 0; k < 64; k++) {
                            if (input[total][k] > max)
                                max = input[total][k];
                            if (input[total][k] < min)
                                min = input[total][k];
                        }
                        if (max != min) {
                            for (int k = 0; k < 64; k++)
                                input[total][k] = (input[total][k] - min) / (max - min);
                        } else if (max != 0) {
                            for (int k = 0; k < 64; k++)
                                input[total][k] /= max;
                        }
                        for (int k = 0; k < 10; k++) {
                            if (k == line.charAt(line.length() - 1) - '0')
                                output[total][k] = 1;
                            else
                                output[total][k] = 0;
                        }

                    }
                }
                // for (double i : input)
                // 	System.out.print(i);
                // System.out.println("");
                //for (int j = 0; j < 10; j++)

                line = reader.readLine();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

//        System.out.println("训练完成，正在测试");

        try {
            System.out.println("正在读取测试集...");
            FileReader ff = new FileReader("digitstest.txt");
            System.out.println("读取完成，正在测试神经网络！");
            @SuppressWarnings("resource")
            BufferedReader rder = new BufferedReader(ff);
            String line = rder.readLine();
            double[][] inputi = new double[1557][64];
            int con = 0;//逗号的个数
            int count = 0;//字符的个数
            int tt = 0;//取出的数字
            int sum = 0;
            int[] tar = new int[1557];
            while (line != null) {
                //处理这一行读出的东西并存进两个数组，完成测试
                while (con < 64) {
                    char tmp = line.charAt(count);//取出的字符
                    if (tmp != ',') {
                        tt = 10 * tt + (tmp - '0');
                        count++;
                    } else if (con != 63) {
                        inputi[sum][con++] = (double) tt;
                        tt = 0;
                        count++;
                    } else {
                        inputi[sum][con++] = (double) tt;
                        count++;
                        tar[sum] = line.charAt(line.length() - 1) - '0';
                        double max = 0;
                        double min = 16;
                        for (int k = 0; k < 64; k++) {
                            if (inputi[sum][k] > max)
                                max = inputi[sum][k];
                            if (inputi[sum][k] < min)
                                min = inputi[sum][k];
                        }
                        if (max != min) {
                            for (int k = 0; k < 64; k++)
                                inputi[sum][k] = (inputi[sum][k] - min) / (max - min);
                        } else if (max != 0) {
                            for (int k = 0; k < 64; k++)
                                inputi[sum][k] /= max;
                        }
                    }
                }
                con = 0;
                count = 0;
                tt = 0;
                sum++;
                line = rder.readLine();
            }
            int max = 1000;
            int cishu = 0;
            //for (int u = 10; u < 200; u++) {
                BP2 bp = new BP2(64, 118, 10);
//                BP bp = new BP(64, 118, 10, 1.7, 1.0);
                //System.out.print(u + ": ");
//                for (int r = 0; r < 600; r++) {
//                    for (int p = 0; p < total; p++)
//                        bp.train(input[p], output[p], 1.7 - 0.3 * r / 600);
//                }
                
                for (int p = 0; p < total; p++){
                	bp.train(input[p], output[p], 1.7);
//                	bp.train(input[p], output[p]);
                }
                
                int correct = 0;
                for (int h = 0; h < 1517; h++) {
                    double outputi[] = bp.test(inputi[h]);

                    int result = 0;
                    for (int j = 0; j < 10; j++) {
                        if (outputi[result] < (int) outputi[j])
                            result = j;
                    }
                    if (result == tar[h])
                        correct++;
                }
                double ratio = (double) correct / sum;
                System.out.println("测试完成！正确率:" + ratio + " " + correct + '/' + sum);
            //}
                /*if(sum == 1517)
                    System.out.println(sum);*/
//                if (correct > max) {
//                    max = correct;
//                    cishu = r;
//                }
        } catch (Exception e) {
            e.printStackTrace();
        }
        //}
//        System.out.println(max);
//        System.out.println(cishu);
        //}
    }

}
