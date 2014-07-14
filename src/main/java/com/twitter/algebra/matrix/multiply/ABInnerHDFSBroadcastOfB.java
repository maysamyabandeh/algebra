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

package com.twitter.algebra.matrix.multiply;

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
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.twitter.algebra.AlgebraCommon;
import com.twitter.algebra.matrix.format.MatrixOutputFormat;

/**
 * Perform A x B matrix multiplication
 * 
 * Approach: Inner-join
 * 
 * Number of jobs: 1 (only map)
 * 
 * Assumption: B is small enough to fit into the memory
 * 
 * Design: load B from HDFS into the mappers' memory. Each mapper perform Ai x B
 * multiplication and generates Bi. There is no need for reducers.
 */
public class ABInnerHDFSBroadcastOfB extends AbstractJob {
  private static final Logger log = LoggerFactory
      .getLogger(ABInnerHDFSBroadcastOfB.class);

  public static final String MATRIXINMEMORY = "matrixInMemory";
  public static final String MATRIXINMEMORYROWS = "memRows";
  public static final String MATRIXINMEMORYCOLS = "memCols";

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(MATRIXINMEMORY, "times",
        "The name of the file that contains the matrix that fits into memory");
    addOption(MATRIXINMEMORYROWS, "r", "Number of rows of the other matrix");
    addOption(MATRIXINMEMORYCOLS, "c", "Number of cols of the other matrix");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    String inMemMatrixFileName = getOption(MATRIXINMEMORY);
    int inMemMatrixNumRows = Integer.parseInt(getOption(MATRIXINMEMORYROWS));
    int inMemMatrixNumCols = Integer.parseInt(getOption(MATRIXINMEMORYCOLS));
    run(getConf(), getInputPath(), inMemMatrixFileName, getOutputPath(),
        inMemMatrixNumRows, inMemMatrixNumCols);
    return 0;
  }

  /**
   * Perform A x B, where A and B are already wrapped in a DistributedRowMatrix
   * object. Refer to {@link ABInnerHDFSBroadcastOfB} for further details.
   * 
   * @param conf
   *          the initial configuration
   * @param A
   *          matrix A
   * @param B
   *          matrix B
   * @param label
   *          the label for the output directory
   * @return AxB wrapped in a DistributedRowMatrix object
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public static DistributedRowMatrix run(Configuration conf,
      DistributedRowMatrix A, DistributedRowMatrix B, String label)
      throws IOException, InterruptedException, ClassNotFoundException {
    log.info("running " + ABInnerHDFSBroadcastOfB.class.getName());
    if (A.numCols() != B.numRows()) {
      throw new CardinalityException(A.numCols(), B.numRows());
    }
    Path outPath = new Path(A.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    ABInnerHDFSBroadcastOfB job = new ABInnerHDFSBroadcastOfB();
    if (!fs.exists(outPath)) {
      job.run(conf, A.getRowPath(), B.getRowPath(), outPath, B.numRows(),
          B.numCols());
    } else {
      log.warn("----------- Skip already exists: " + outPath);
    }
    DistributedRowMatrix distRes = new DistributedRowMatrix(outPath,
        A.getOutputTempPath(), A.numRows(), B.numCols());
    distRes.setConf(conf);
    return distRes;
  }

  /**
   * Perform A x B, where A and B refer to the paths that contain matrices in
   * {@link SequenceFileInputFormat} Refer to {@link ABInnerHDFSBroadcastOfB}
   * for further details.
   * 
   * @param conf
   *          the initial configuration
   * @param matrixInputPath
   *          path to matrix A
   * @param inMemMatrixDir
   *          path to matrix B (must be small enough to fit into memory)
   * @param matrixOutputPath
   *          path to which AxB will be written
   * @param inMemMatrixNumRows
   *          B rows
   * @param inMemMatrixNumCols
   *          B cols
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public void run(Configuration conf, Path matrixInputPath,
      Path inMemMatrixDir, Path matrixOutputPath, int inMemMatrixNumRows,
      int inMemMatrixNumCols) throws IOException, InterruptedException,
      ClassNotFoundException {
    run(conf, matrixInputPath, inMemMatrixDir.toString(), matrixOutputPath,
        inMemMatrixNumRows, inMemMatrixNumCols);
  }

  /**
   * Perform A x B, where A and B refer to the paths that contain matrices in
   * {@link SequenceFileInputFormat} Refer to {@link ABInnerHDFSBroadcastOfB}
   * for further details.
   * 
   * @param conf
   *          the initial configuration
   * @param matrixInputPath
   *          path to matrix A
   * @param inMemMatrixDir
   *          path to matrix B (must be small enough to fit into memory)
   * @param matrixOutputPath
   *          path to which AxB will be written
   * @param inMemMatrixNumRows
   *          B rows
   * @param inMemMatrixNumCols
   *          B cols
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public void run(Configuration conf, Path matrixInputPath,
      String inMemMatrixDir, Path matrixOutputPath, int inMemMatrixNumRows,
      int inMemMatrixNumCols) throws IOException, InterruptedException,
      ClassNotFoundException {
    conf.set(MATRIXINMEMORY, inMemMatrixDir);
    conf.setInt(MATRIXINMEMORYROWS, inMemMatrixNumRows);
    conf.setInt(MATRIXINMEMORYCOLS, inMemMatrixNumCols);
    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(ABInnerHDFSBroadcastOfB.class);
    job.setJobName(ABInnerHDFSBroadcastOfB.class.getSimpleName());
    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);
    matrixInputPath = fs.makeQualified(matrixInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    FileInputFormat.addInputPath(job, matrixInputPath);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setMapperClass(MyMapper.class);
    
    job.setNumReduceTasks(0);
    job.setOutputFormatClass(MatrixOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);

    // since we do not use reducer, to get total order, the map output files has
    // to be renamed after this function returns: {@link AlgebraCommon#fixPartitioningProblem}
    job.submit();
    boolean res = job.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed!");
  }

  public static class MyMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    private DenseMatrix inMemMatrix;
    private DenseVector resVector;
    private VectorWritable vectorWritable = new VectorWritable();

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      Path inMemMatrixPath = new Path(conf.get(MATRIXINMEMORY));
      int inMemMatrixNumRows = conf.getInt(MATRIXINMEMORYROWS, 0);
      int inMemMatrixNumCols = conf.getInt(MATRIXINMEMORYCOLS, 0);
      //TODO: Add support for in-memory sparse matrix
      inMemMatrix = AlgebraCommon.mapDirToDenseMatrix(inMemMatrixPath,
          inMemMatrixNumRows, inMemMatrixNumCols, conf);
    }

    @Override
    public void map(IntWritable r, VectorWritable v, Context context)
        throws IOException, InterruptedException {
      Vector row = v.get();
      if (resVector == null)
        resVector = new DenseVector(inMemMatrix.numCols());
      AlgebraCommon.vectorTimesMatrix(row, inMemMatrix, resVector);
      vectorWritable.set(resVector);
      context.write(r, vectorWritable);
    }
  }

}
