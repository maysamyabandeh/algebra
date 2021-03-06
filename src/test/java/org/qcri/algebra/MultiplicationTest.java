/*
Copyright 2014 Twitter, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package org.qcri.algebra;

import java.io.IOException;

import junit.framework.Assert;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.junit.Before;
import org.junit.Test;

import com.twitter.algebra.AlgebraCommon;
import com.twitter.algebra.TransposeJob;
import com.twitter.algebra.matrix.multiply.ABInnerHDFSBroadcastOfB;
import com.twitter.algebra.matrix.multiply.ABOuterHDFSBroadcastOfA;
import com.twitter.algebra.matrix.multiply.AtBOuterStaticMapsideJoinJob;
import com.twitter.algebra.matrix.multiply.AtB_DMJ;
import com.twitter.algebra.nmf.ColPartitionJob;
import com.twitter.algebra.nmf.CompositeDMJ;
import com.twitter.algebra.nmf.NMFCommon;
import com.twitter.algebra.nmf.RowSquareSumJob;
import com.twitter.algebra.nmf.XtXJob;

public class MultiplicationTest extends Assert {
  public static final double EPSILON = 0.000001;

//  private final double[][] inputVectorsA = { { 1, 2 }, { 3, 4 } };
//  private final double[][] inputVectorsA2 = { { 1, 2 }, { 3, 4 } };
//  private final double[][] inputVectorsAsquare = { { 1, 2 }, { 3, 4 } };
//   private final double[][] inputVectorsB = { { 4, 5 }, { 6, 7 } };
  private final double[][] inputVectorsA = { { -0.49, 0, 3, -13.3 },
      { 0, -0.48, 0.45, 3.2 }, { 0.13, -0.06, 4, -0.00003 } };
  private final double[][] inputVectorsAsquare = { { -0.49, 0, 3, -13.3 },
      { 0, -0.48, 0.45, 3.2 }, { 0, -0.48, 0.45, 3.2 }, { 0.13, -0.06, 4, -0.00003 } };
  private final double[][] inputVectorsA2 = { { -0.49, 2.1, 0, -13.3 },
      { -1.3, -0.48, 0, 3.2 }, { 0.13, -0.06, 0, -0.00003 } };
  private final double[][] inputVectorsB = { { -0.49, 0 }, { 0, -0.48 },
      { 0, 0.71 }, { 4, -0.00003 } };
  private final int rowsA = inputVectorsA.length;
  private final int rowsB = inputVectorsB.length;
  private final int colsA = inputVectorsA[0].length;
  private final int colsB = inputVectorsB[0].length;
  private final double[][] dotVectors = new double[rowsA][colsA];
  private final double[][] compositeVectors = new double[rowsA][colsA];
  private final double[][] productVectors = new double[rowsA][colsB];
  private final double[][] ataVectors = new double[colsA][colsA];
  Configuration conf;

  private Path output;
  private Path tmp;

  private Path aDensePath;
  private Path bDensePath;
  private Path atDensePath;
  private Path aSparsePath;
  private Path bSparsePath;
  private Path atSparsePath;
  private Path a2DensePath;
  private Path a2SparsePath;
  private Path aSquareDensePath;
  private Path aSquareSparsePath;

  private double[][] aTranspose;

  @Test
  public void testMath() {
    int id = 280;
    System.out.println(id / (double) 1000);
    System.out.println(id / (double) 1000 * 100);

    int res = (int) (id * 100 / (double) 1000);
    System.out.println(res);
    id = 290;
    System.out.println(id / (double) 1000);
    System.out.println(id / (double) 1000 * 100);
    res = (int) (id * 100 / (double) 1000);
    System.out.println(res);
    
    int row = 2001231;
    int numPartitions = 1500;
    int totalKeys = 37200000;
    System.out.println( (int) ((row * (long) numPartitions / (double) totalKeys)) );
    System.out.println(  ((row * (long) numPartitions / (double) totalKeys)) );
    System.out.println(  ((row * (long) numPartitions )) );

  }
  
  @Before
  public void setup() throws Exception {
    NMFCommon.DEFAULT_REDUCESPLOTS = 2;
    dot(inputVectorsA, inputVectorsA2, dotVectors);
    composite(inputVectorsA, inputVectorsA2, inputVectorsAsquare, compositeVectors);
    product(inputVectorsA, inputVectorsB, productVectors);
    ata(inputVectorsA, ataVectors);
    conf = new Configuration();
    conf.set("mapreduce.job.tracker", "local");
    conf.set("fs.default.name", "file:///");
    long currTime = System.currentTimeMillis();
    Path testbed = new Path("/tmp/" + currTime);
    output = new Path(testbed, "output");
    tmp = new Path(testbed, "tmp");
    FileSystem fs;
    try {
      fs = FileSystem.get(output.toUri(), conf);
      fs.mkdirs(output);
      fs.mkdirs(tmp);
      fs.deleteOnExit(testbed);
    } catch (IOException e) {
      e.printStackTrace();
      Assert.fail("Error in creating output direcoty " + output);
      return;
    }
    aTranspose = transpose(inputVectorsA);
    aDensePath =
        AlgebraCommon.toDenseMapDir(inputVectorsA, tmp, tmp, "matrixADense")
            .getRowPath();
    aSquareDensePath =
        AlgebraCommon.toDenseMapDir(inputVectorsAsquare, tmp, tmp, "matrixASqaureDense")
            .getRowPath();
    a2DensePath =
        AlgebraCommon.toDenseMapDir(inputVectorsA2, tmp, tmp, "matrixA2Dense")
            .getRowPath();
    bDensePath =
        AlgebraCommon.toDenseMapDir(inputVectorsB, tmp, tmp, "matrixBDense")
            .getRowPath();
    atDensePath =
        AlgebraCommon.toDenseMapDir(aTranspose, tmp, tmp, "matrixAtDense")
            .getRowPath();
    aSparsePath =
        AlgebraCommon.toSparseMapDir(inputVectorsA, tmp, tmp, "matrixASparse")
            .getRowPath();
    aSquareSparsePath =
        AlgebraCommon.toSparseMapDir(inputVectorsAsquare, tmp, tmp, "matrixASqaureSparse")
            .getRowPath();
    a2SparsePath =
        AlgebraCommon.toSparseMapDir(inputVectorsA2, tmp, tmp, "matrixA2Sparse")
            .getRowPath();
    bSparsePath =
        AlgebraCommon.toSparseMapDir(inputVectorsB, tmp, tmp, "matrixBSparse")
            .getRowPath();
    atSparsePath =
        AlgebraCommon.toSparseMapDir(aTranspose, tmp, tmp, "matrixAtSparse")
            .getRowPath();
  }

  @Test
  public void testTransposeJob() throws Exception {
    testTransposeJob(aDensePath, "Dense");
    System.gc();// otherwise my jvm does not automatically free the memory!
    testTransposeJob(aSparsePath, "Sparse");
  }

  public void testTransposeJob(Path aPath, String label) throws Exception {
    DistributedRowMatrix a =
        new DistributedRowMatrix(aPath, tmp, rowsA, colsA);
    a.setConf(conf);
    DistributedRowMatrix distAt =
        TransposeJob.transpose(a, conf, "At"+label);
    verifyTranspose(distAt.getRowPath());
  }

  @Test
  public void testXtXJob() throws Exception {
    testXtXJob(aDensePath, "Dense");
    System.gc();// otherwise my jvm does not automatically free the memory!
    testXtXJob(aSparsePath, "Sparse");
  }

  public void testXtXJob(Path aPath, String label) throws Exception {

    DistributedRowMatrix a =
        new DistributedRowMatrix(aPath, tmp, rowsA, colsA);
    a.setConf(conf);
    DistributedRowMatrix distAtA =
        new XtXJob().computeXtX(a, tmp, conf, XtXJob.class.getName() + label);
    verifyAtA(distAtA.getRowPath());
  }
  
  @Test
  public void tesSumJob() throws Exception {
    tesSumJob(aDensePath, "Dense");
    tesSumJob(aSparsePath, "Sparse");
  }

  public void tesSumJob(Path aPath, String label) throws Exception {
    DistributedRowMatrix a =
        new DistributedRowMatrix(aPath, tmp, rowsA, colsA);
    a.setConf(conf);
    Path outPath = new Path(output, RowSquareSumJob.class.getName() + label);
    new RowSquareSumJob().run(conf, aPath, outPath, a.numRows());
    verifySquareSum(outPath);
  }

  @Test
  public void testCompositeDMJ() throws Exception {
    testCompositeDMJ(aDensePath, a2DensePath, aSquareDensePath, "Dense");
    testCompositeDMJ(aSparsePath, a2SparsePath, aSquareSparsePath, "Sparse");
  }

  public void testCompositeDMJ(Path aPath, Path bPath, Path aSquarePath,
      String label) throws Exception {
    CompositeDMJ job = new CompositeDMJ();
    Path outPath =
        new Path(output, CompositeDMJ.class.getName() + label);
    job.run(conf, aPath, bPath, outPath, rowsA, aSquarePath, colsA, colsA, 0, 0);
    verifyCompositeDMJ(outPath);
  }

  @Test
  public void testAtBOuterProductMapsideJoinJob() throws Exception {
    testAtBOuterProductMapsideJoinJob(atDensePath, bDensePath, "Dense");
    System.gc();// otherwise my jvm does not automatically free the memory!
    testAtBOuterProductMapsideJoinJob(atSparsePath, bSparsePath, "Sparse");
  }

  public void testAtBOuterProductMapsideJoinJob(Path atPath, Path bPath,
      String label) throws Exception {
    AtBOuterStaticMapsideJoinJob job = new AtBOuterStaticMapsideJoinJob();
    Path outPath =
        new Path(output, AtBOuterStaticMapsideJoinJob.class.getName() + label);
    job.run(conf, atPath, bPath, outPath, colsB);
    verifyProduct(outPath);
  }

  @Test
  public void testDMJ() throws Exception {
    //without column partition
    testDMJ(atDensePath, bDensePath, "Dense", 1, 1, true);
    testDMJ(atSparsePath, bSparsePath, "Sparse", 1, 1, true);
    testDMJ(atDensePath, bDensePath, "Dense", 1, 1, false);
    testDMJ(atSparsePath, bSparsePath, "Sparse", 1, 1, false);

    //with column partition At
    testDMJ(atDensePath, bDensePath, "Dense", 2, 1, true);
    testDMJ(atSparsePath, bSparsePath, "Sparse", 2, 1, true);
    testDMJ(atDensePath, bDensePath, "Dense", 2, 1, false);
    testDMJ(atSparsePath, bSparsePath, "Sparse", 2, 1, false);

    //with column partition B
    testDMJ(atDensePath, bDensePath, "Dense", 1, 2, true);
    testDMJ(atSparsePath, bSparsePath, "Sparse", 1, 2, true);
    testDMJ(atDensePath, bDensePath, "Dense", 1, 2, false);
    testDMJ(atSparsePath, bSparsePath, "Sparse", 1, 2, false);
    
    //with column partition larger than available columns At
    testDMJ(atDensePath, bDensePath, "Dense", rowsA + 1, 1, true);
    testDMJ(atSparsePath, bSparsePath, "Sparse", rowsA + 1, 1, true);
    testDMJ(atDensePath, bDensePath, "Dense", rowsA + 1, 1, false);
    testDMJ(atSparsePath, bSparsePath, "Sparse", rowsA + 1, 1, false);
    
    //with column partition larger than available columns B
    testDMJ(atDensePath, bDensePath, "Dense", 1, colsB + 1, true);
    testDMJ(atSparsePath, bSparsePath, "Sparse", 1, colsB + 1, true);
    testDMJ(atDensePath, bDensePath, "Dense", 1, colsB + 1, false);
    testDMJ(atSparsePath, bSparsePath, "Sparse", 1, colsB + 1, false);

    //with column partition, #reducers > col partitions At
    int nReducers = 2 * 3;
    NMFCommon.DEFAULT_REDUCESPLOTS = nReducers;
    testDMJ(atDensePath, bDensePath, "Dense" + nReducers + "_At" , 2, 1, true);
    testDMJ(atSparsePath, bSparsePath, "Sparse" + nReducers + "_At" , 2, 1, true);
    testDMJ(atDensePath, bDensePath, "Dense" + nReducers + "_At" , 2, 1, false);
    testDMJ(atSparsePath, bSparsePath, "Sparse" + nReducers + "_At" , 2, 1, false);
    
    //with column partition, #reducers > col partitions B
    nReducers = 2 * 3;
    NMFCommon.DEFAULT_REDUCESPLOTS = nReducers;
    testDMJ(atDensePath, bDensePath, "Dense" + nReducers + "_B" , 1, 2, true);
    testDMJ(atSparsePath, bSparsePath, "Sparse" + nReducers + "_B" , 1, 2, true);
    testDMJ(atDensePath, bDensePath, "Dense" + nReducers + "_B" , 1, 2, false);
    testDMJ(atSparsePath, bSparsePath, "Sparse" + nReducers + "_B" , 1, 2, false);
    
    // with column partition, #reducers > col partitions At, but
    // #reducers/#colPart is not a natural number
    nReducers = 2 + 1;
    NMFCommon.DEFAULT_REDUCESPLOTS = nReducers;
    testDMJ(atDensePath, bDensePath, "Dense" + nReducers + "_At" , 2, 1, true);
    testDMJ(atSparsePath, bSparsePath, "Sparse" + nReducers + "_At" , 2, 1, true);
    testDMJ(atDensePath, bDensePath, "Dense" + nReducers + "_At" , 2, 1, false);
    testDMJ(atSparsePath, bSparsePath, "Sparse" + nReducers + "_At" , 2, 1, false);
    
    // with column partition, #reducers > col partitions B, but
    // #reducers/#colPart is not a natural number
    nReducers = 2 + 1;
    NMFCommon.DEFAULT_REDUCESPLOTS = nReducers;
    testDMJ(atDensePath, bDensePath, "Dense" + nReducers + "_B" , 1, 2, true);
    testDMJ(atSparsePath, bSparsePath, "Sparse" + nReducers + "_B" , 1, 2, true);
    testDMJ(atDensePath, bDensePath, "Dense" + nReducers + "_B" , 1, 2, false);
    testDMJ(atSparsePath, bSparsePath, "Sparse" + nReducers + "_B" , 1, 2, false);
}

  public void testDMJ(Path atPath, Path bPath, String label,
      int nColPartitionsOfAt, int nColPartitionsOfB, boolean useInMemCombiner)
      throws Exception {
    if (nColPartitionsOfB != 1 && nColPartitionsOfAt != 1)
      throw new Exception("DMJ input error: only one of At and B can be column partitioned!");
    AtB_DMJ job = new AtB_DMJ();
    label = label + nColPartitionsOfAt + "-" + nColPartitionsOfB + "-" + useInMemCombiner;
    Path outPath =
        new Path(output, AtB_DMJ.class.getName() + label);
    if (nColPartitionsOfAt != 1) 
      atPath = colPartition(atPath, colsA, rowsA, nColPartitionsOfAt, label);
    else if (nColPartitionsOfB != 1) 
      bPath = colPartition(bPath, rowsB, colsB, nColPartitionsOfB, label);
    job.run(conf, atPath, bPath, outPath, rowsA, colsB, nColPartitionsOfAt, nColPartitionsOfB, useInMemCombiner);
    verifyProduct(outPath);
  }
  
  Path colPartition(Path inPath, int rows, int cols, int nColPartitions,
      String label) throws IOException, InterruptedException,
      ClassNotFoundException {
    DistributedRowMatrix distIn =
        new DistributedRowMatrix(inPath, tmp, rows, cols);
    distIn.setConf(conf);
    DistributedRowMatrix distOut =
        ColPartitionJob.partition(distIn, conf, label, nColPartitions);
    return distOut.getRowPath();
  }

  @Test
  public void testABInnerHDFSBroadcastOfB() throws Exception {
    testABInnerHDFSBroadcastOfB(aDensePath, bDensePath, "Dense");
    System.gc();// otherwise my jvm does not automatically free the memory!
    testABInnerHDFSBroadcastOfB(aSparsePath, bSparsePath, "Sparse");
  }

  public void testABInnerHDFSBroadcastOfB(Path aPath, Path bPath, String label)
      throws Exception {
    ABInnerHDFSBroadcastOfB job = new ABInnerHDFSBroadcastOfB();
    Path outPath =
        new Path(output, ABInnerHDFSBroadcastOfB.class.getName() + label);
    job.run(conf, aPath, bPath, outPath, rowsB, colsB);
    verifyProduct(outPath);
  }

  @Test
  public void testABOuterHDFSBroadcastOfA() throws Exception {
    testABOuterHDFSBroadcastOfA(aDensePath, bDensePath, "Dense");
    System.gc();// otherwise my jvm does not automatically free the memory!
    testABOuterHDFSBroadcastOfA(aSparsePath, bSparsePath, "Sparse");
  }

  public void testABOuterHDFSBroadcastOfA(Path aPath, Path bPath, String label)
      throws Exception {
    ABOuterHDFSBroadcastOfA job = new ABOuterHDFSBroadcastOfA();
    Path outPath =
        new Path(output, ABOuterHDFSBroadcastOfA.class.getName() + label);
    job.run(conf, aPath, bPath, outPath, rowsA, colsA);
    verifyProduct(outPath);
  }

  private double[][] transpose(double[][] inputVectors) {
    int rows = inputVectors.length;
    int cols = inputVectors[0].length;
    double[][] transpose = new double[cols][rows];
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++)
        transpose[c][r] = inputVectors[r][c];
    return transpose;
  }

  void verifyProduct(Path productPath) throws IOException {
    Matrix productMatrix =
        AlgebraCommon.mapDirToDenseMatrix(productPath, rowsA, colsB, conf);
    for (int r = 0; r < productMatrix.numRows(); r++) {
      for (int c = 0; c < productMatrix.numCols(); c++)
        Assert.assertEquals("The product[" + r + "][" + c + "] is incorrect: ",
            productVectors[r][c], productMatrix.get(r, c), EPSILON);
    }
  }

  void verifyCompositeDMJ(Path productPath) throws IOException {
    Matrix compositeMatrix =
        AlgebraCommon.mapDirToDenseMatrix(productPath, rowsA, colsA, conf);
    for (int r = 0; r < compositeMatrix.numRows(); r++) {
      for (int c = 0; c < compositeMatrix.numCols(); c++)
        Assert.assertEquals("The composite[" + r + "][" + c + "] is incorrect: ",
            compositeVectors[r][c], compositeMatrix.get(r, c), EPSILON);
    }
  }

  void verifyTranspose(Path atPath) throws IOException {
    Matrix atMatrix =
        AlgebraCommon.mapDirToDenseMatrix(atPath, colsA, rowsA, conf);
    for (int r = 0; r < atMatrix.numRows(); r++) {
      for (int c = 0; c < atMatrix.numCols(); c++)
        Assert.assertEquals("The ata[" + r + "][" + c + "] is incorrect: ",
            inputVectorsA[c][r], atMatrix.get(r, c), EPSILON);
    }
  }

  void verifyAtA(Path ataPath) throws IOException {
    Matrix ataMatrix =
        AlgebraCommon.mapDirToDenseMatrix(ataPath, colsA, colsA, conf);
    for (int r = 0; r < ataMatrix.numRows(); r++) {
      for (int c = 0; c < ataMatrix.numCols(); c++)
        Assert.assertEquals("The ata[" + r + "][" + c + "] is incorrect: ",
            ataVectors[r][c], ataMatrix.get(r, c), EPSILON);
    }
  }

  void verifySquareSum(Path sumPath) throws IOException {
    Vector sumVec =
        AlgebraCommon.mapDirToSparseVector(sumPath, 1, colsA, conf);
    double[][] vectorsA = inputVectorsA; 
    for (int r = 0; r < vectorsA.length; r++) {
      double sum = 0;
      for (int c = 0; c < vectorsA[0].length; c++)
        sum += vectorsA[r][c] * vectorsA[r][c];
      Assert.assertEquals("The sum of a[" + r + "][*] is incorrect: ",
          sum, sumVec.get(r), EPSILON);
    }
  }

  private void ata(double[][] vectorsA, double[][] resVectors) {
    for (int r = 0; r < vectorsA.length; r++)
      for (int c = 0; c < vectorsA[0].length; c++) {
        for (int i = 0; i < vectorsA[0].length; i++)
          resVectors[c][i] += vectorsA[r][c] * vectorsA[r][i];
      }
  }

  private void product(double[][] vectorsA, double[][] vectorsB,
      double[][] resVectors) {
    for (int r = 0; r < vectorsA.length; r++)
      for (int c = 0; c < vectorsB[0].length; c++) {
        for (int i = 0; i < vectorsB.length; i++)
          resVectors[r][c] += vectorsA[r][i] * vectorsB[i][c];
      }
  }

  private void composite(double[][] vectorsA, double[][] vectorsA2, double[][] vectorsASquare,
      double[][] resVectors) {
    double[][] multiplyRes = vectorsA.clone();
    for (int i = 0; i < vectorsA.length; i++)
      multiplyRes[i] = new double[vectorsA[0].length];
    product(vectorsA, vectorsASquare, multiplyRes);
    for (int r = 0; r < vectorsA.length; r++)
      for (int c = 0; c < vectorsA[0].length; c++) {
          resVectors[r][c] = vectorsA[r][c] * vectorsA2[r][c] / multiplyRes[r][c];
      }
  }

  private void dot(double[][] vectorsA, double[][] vectorsA2,
      double[][] resVectors) {
    for (int r = 0; r < vectorsA.length; r++)
      for (int c = 0; c < vectorsA[0].length; c++) {
          resVectors[r][c] = vectorsA[r][c] * vectorsA2[r][c];
      }
  }

}
