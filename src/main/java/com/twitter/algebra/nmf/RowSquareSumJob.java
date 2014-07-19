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

package com.twitter.algebra.nmf;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.twitter.algebra.AlgebraCommon;
import com.twitter.algebra.MergeVectorsReducer;
import com.twitter.algebra.matrix.format.MatrixOutputFormat;

/**
 * Sum up the rows of a matrix
 * 
 * @author myabandeh
 * 
 */
public class RowSquareSumJob extends AbstractJob {
  private static final Logger log = LoggerFactory
      .getLogger(RowSquareSumJob.class);

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOption("numRows", "nr", "Number of rows of the input matrix");
    addOption("numCols", "nc", "Number of columns of the input matrix");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    int numRows = Integer.parseInt(getOption("numRows"));
    int numCols = Integer.parseInt(getOption("numCols"));

    DistributedRowMatrix matrix =
        new DistributedRowMatrix(getInputPath(), getTempPath(), numRows,
            numCols);
    matrix.setConf(new Configuration(getConf()));
    RowSquareSumJob.run(getConf(), matrix, "combined-" + getInputPath());
    return 0;
  }

  /**
   * Returns the path to the vector that contains the sum of the rows of A
   * @param conf
   * @param A
   * @param label
   * @return
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public static Path run(Configuration conf, DistributedRowMatrix A, String label)
      throws IOException, InterruptedException, ClassNotFoundException {
    log.info("running " + RowSquareSumJob.class.getName());
    Path outPath = new Path(A.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    RowSquareSumJob job = new RowSquareSumJob();
    if (!fs.exists(outPath)) {
      job.run(conf, A.getRowPath(), outPath, A.numRows());
    } else {
      log.warn("----------- Skip already exists: " + outPath);
    }
    Matrix centRes = AlgebraCommon.mapDirToSparseMatrix(outPath, A.numRows(),
        A.numCols(), conf);
    Vector resVec = centRes.viewRow(0);
    System.out.println("Sum of the rows of " + A.getRowPath());
    System.out.println(resVec);
    return outPath;
  }

  public void run(Configuration conf, Path matrixInputPath,
      Path matrixOutputPath, int aRows) throws IOException, InterruptedException,
      ClassNotFoundException {
    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(RowSquareSumJob.class);
    job.setJobName(RowSquareSumJob.class.getSimpleName() + "-" + matrixOutputPath.getName());
    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);
    matrixInputPath = fs.makeQualified(matrixInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    FileInputFormat.addInputPath(job, matrixInputPath);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    
    int numReducers = 1;
    job.setNumReduceTasks(numReducers);
    
    job.setOutputFormatClass(MatrixOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    
    job.setMapperClass(SumMapper.class);
    job.setCombinerClass(MergeVectorsReducer.class);
    job.setReducerClass(MergeVectorsReducer.class);

//    RowPartitioner.setPartitioner(job, RowPartitioner.IntRowPartitioner.class,
//        aRows);
    job.submit();
    boolean res = job.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed!");
  }

  public static class SumMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    IntWritable iw = new IntWritable(0);//single key
    VectorWritable vw = new VectorWritable();
    @Override
    public void map(IntWritable key, VectorWritable value, Context context) throws IOException,
        InterruptedException {
      Vector inVec = value.get();
      Double sqaureSum = inVec.aggregate(Functions.PLUS, Functions.pow(2));
      Vector sumVec = new SequentialAccessSparseVector(inVec.size());
      sumVec.set(key.get(), sqaureSum);
      vw.set(sumVec);
      context.write(iw, vw);
    }
  }
}

























