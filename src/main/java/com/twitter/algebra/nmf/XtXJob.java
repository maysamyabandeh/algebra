/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.twitter.algebra.nmf;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.myabandeh.algebra.matrix.format.MatrixOutputFormat;
import com.myabandeh.algebra.matrix.format.RowPartitioner;

/**
 * Xt * X
 * @author maysam yabandeh
 */
public class XtXJob extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(XtXJob.class);
  public static final String MATRIXCOLS = "matrix.num.cols";
//  public static final String XMPATH = "xmPath";

  @Override
  public int run(String[] strings) throws Exception {
    throw new Exception("Unimplemented");
  }

  public DistributedRowMatrix computeXtX(DistributedRowMatrix xMatrix,
      Path tmpPath, Configuration conf, String id)
      throws IOException, InterruptedException, ClassNotFoundException {
    return computeXtX(xMatrix, null, tmpPath, conf, id);
  }

  public DistributedRowMatrix computeXtX(DistributedRowMatrix xMatrix,
      Vector xm, Path tmpPath, Configuration conf, String id)
      throws IOException, InterruptedException, ClassNotFoundException {
    Path outPath = new Path(tmpPath, "XtX-" + id);
//    Path xmPath =
//        AlgebraCommon.toDistributedVector(xm, tmpPath, "xm-XtXJob" + id, conf);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    if (!fs.exists(outPath)) {
      run(conf, xMatrix.getRowPath(), xMatrix.numCols(), null,
          outPath);
    } else {
      log.warn("----------- Skip XtXjob - already exists: " + outPath);
    }

    DistributedRowMatrix xtx =
        new DistributedRowMatrix(outPath, tmpPath, xMatrix.numCols(),
            xMatrix.numCols());
    xtx.setConf(conf);
    return xtx;
  }

  public void run(Configuration conf, Path matrixInputPath, int numCols,
      String xmPath, Path matrixOutputPath) throws IOException,
      InterruptedException, ClassNotFoundException {
    conf.setInt(MATRIXCOLS, numCols);
//    conf.set(XMPATH, xmPath);
    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);
    NMFCommon.setNumberOfMapSlots(conf, fs, new Path[] {matrixInputPath}, "xtx");

    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJobName("XtXJob-" + matrixOutputPath.getName());
    job.setJarByClass(XtXJob.class);
    matrixInputPath = fs.makeQualified(matrixInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);
    FileInputFormat.addInputPath(job, matrixInputPath);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setMapperClass(MyMapper.class);
    job.setReducerClass(MyReducer.class);

    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    int numReducers = NMFCommon.getNumberOfReduceSlots(conf, "xtx");
    job.setNumReduceTasks(numReducers);
    // ensures total order (when used with {@link MatrixOutputFormat}),
    RowPartitioner.setPartitioner(job, RowPartitioner.IntRowPartitioner.class,
        numCols);

    job.setOutputFormatClass(MatrixOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);

    job.submit();
    job.waitForCompletion(true);
  }

  public static class MyMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
//    private Vector xm;
    int numCols;
    // developing variables
    private DenseMatrix xtxMatrix;

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      numCols = conf.getInt(MATRIXCOLS, 0);
//      Path xmPath = new Path(conf.get(XMPATH));
//      try {
//        xm = AlgebraCommon.toDenseVector(xmPath, conf);
//      } catch (IOException e) {
//        e.printStackTrace();
//      }
      xtxMatrix = new DenseMatrix(numCols, numCols);
    }

    /**
     * Perform in-memory vector multiplication xtx += xi' * xi
     */
    @Override
    public void map(IntWritable r, VectorWritable v, Context context)
        throws IOException, InterruptedException {
      Vector xi = v.get();
      xtx(xi, xtxMatrix);
    }

    @Override
    public void cleanup(Context context) throws InterruptedException,
        IOException {
      IntWritable key = new IntWritable();
      VectorWritable outVector = new VectorWritable();
      for (int i = 0; i < xtxMatrix.numRows(); i++) {
        key.set(i);
        outVector.set(xtxMatrix.viewRow(i));
        context.write(key, outVector);
      }
    }

    private void xtx(Vector xi,DenseMatrix resMatrix) {
      Iterator<Vector.Element> rowIterator = xi.nonZeroes().iterator();
      while (rowIterator.hasNext()) {
        Vector.Element e = rowIterator.next();
        int xRow = e.index();
        double xScale = e.get(); // - xm[i]
        Iterator<Vector.Element> colIterator = xi.nonZeroes().iterator();
        while (colIterator.hasNext()) {
          Vector.Element colElmnt = colIterator.next();
          int xCol = colElmnt.index();
          double centeredValue = colElmnt.get(); // - xm[i]
          double currValue = resMatrix.getQuick(xRow, xCol);
          currValue += centeredValue * xScale;
          resMatrix.setQuick(xRow, xCol, currValue);
        }
      }
    }
  }

  public static class MyReducer extends
      Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    IntWritable iw = new IntWritable();
    VectorWritable vw = new VectorWritable();

    @Override
    public void reduce(IntWritable key, Iterable<VectorWritable> vectors,
        Context context) throws IOException, InterruptedException {
      Iterator<VectorWritable> it = vectors.iterator();
      if (!it.hasNext()) {
        return;
      }
      // Reduce XtX
      Vector accumulator = it.next().get();
      while (it.hasNext()) {
        Vector row = it.next().get();
        accumulator.assign(row, Functions.PLUS);
      }
      vw.set(accumulator);
      context.write(key, vw);
    }
  }

}
