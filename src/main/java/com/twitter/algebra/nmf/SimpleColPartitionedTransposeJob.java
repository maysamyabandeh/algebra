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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang.builder.HashCodeBuilder;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.LazyOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.ComparisonChain;
import com.myabandeh.algebra.matrix.format.MatrixOutputFormat;
import com.myabandeh.algebra.matrix.format.RowPartitioner;
import com.twitter.algebra.nmf.NMFCommon;

/**
 * Transpose a matrix A into At while partitioning the columns of At. It is
 * necessary when A has many rows (At has many columns). It also allows
 * {@link DMJ} to split the multiplication into multiple
 * jobs.
 * 
 * Note: there is a tradeoff for the number of column partitions. Many column
 * partitions hurt the performance by redundant remote reading of the other
 * matrix.
 */
public class SimpleColPartitionedTransposeJob extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(SimpleColPartitionedTransposeJob.class);

  public static final String NUM_ORIG_ROWS_KEY = "SparseRowMatrix.numRows";
  public static final String NUM_COL_PARTITIONS = "TransposeMatrix.numColPartitions";
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new SimpleColPartitionedTransposeJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOption("numRows", "nr", "Number of rows of the input matrix");
    addOption("numCols", "nc", "Number of columns of the input matrix");
    addOption("resColPart", "cp", "Number of column partitions of the output matrix");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    int numRows = Integer.parseInt(getOption("numRows"));
    int numCols = Integer.parseInt(getOption("numCols"));
    int numColPartitions = Integer.parseInt(getOption("resColPart"));

    DistributedRowMatrix matrix = new DistributedRowMatrix(getInputPath(),
        getTempPath(), numRows, numCols);
    matrix.setConf(new Configuration(getConf()));
    transpose(matrix, getConf(), "Transpose-" + getInputPath(), 1, numColPartitions);

    return 0;
  }

  /**
   * Perform transpose of A, where A is already wrapped in a DistributedRowMatrix
   * object. Refer to {@link SimpleColPartitionedTransposeJob} for further details.
   * 
   * @param conf
   *          the initial configuration
   * @param distM
   *          input matrix A
   * @param label
   *          the label for the output directory
   * @param numberOfJobs
   *          the hint for the desired number of parallel jobs
   * @param numColPartitions 
   *          the hint for the desired number of column partitions
   * @return At wrapped in a DistributedRowMatrix object
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public static DistributedRowMatrix transpose(DistributedRowMatrix distM,
      Configuration conf, String label, int numberOfJobs, int numColPartitions) 
          throws IOException, InterruptedException, ClassNotFoundException {
    Path outputPath = new Path(distM.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outputPath.toUri(), conf);
    if (!fs.exists(outputPath)) {
      runJobsInParallel(conf, distM.getRowPath(), outputPath, distM.numRows(), distM.numCols(),
          numberOfJobs, numColPartitions);
    } else {
      log.warn("----------- Skip already exists: " + outputPath);
    }
    DistributedRowMatrix m = new DistributedRowMatrix(outputPath,
        distM.getOutputTempPath(), distM.numCols(), distM.numRows());
    m.setConf(conf);
    return m;
  }
  
  public static class ExcludeMetaFilesFilter implements PathFilter {
    @Override
    public boolean accept(Path path) {
      String name = path.getName();
      return !name.startsWith(".") && !name.startsWith("_");
    }
  }
    
  /**
   * Split a big transpose job into multiple smaller jobs. Each job should be
   * more efficient as it puts less load on the reducers. We also can run the
   * jobs in parallel.
   * 
   * The maximum number of jobs is the number of input files.
   * 
   * @param conf
   * @param matrixInputPath
   *          the path to input matrix A
   * @param matrixOutputPath
   *          the output matrix path (At)
   * @param numInputRows
   *          rows
   * @param numInputCols
   *          cols
   * @param numberOfJobs
   *          the hint for the desired number of parallel jobs
   * @param numColPartitions
   *          the hint for the desired number of column partitions
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */  
  @SuppressWarnings("deprecation")
  private static void runJobsInParallel(Configuration conf,
      Path matrixInputPath, Path matrixOutputPath, int numInputRows,
      int numInputCols, int numberOfJobs, int numColPartitions) throws IOException,
      InterruptedException, ClassNotFoundException {
    FileSystem fs = FileSystem.get(matrixOutputPath.toUri(), conf);
    FileStatus[] files = fs.listStatus(matrixInputPath,
        new ExcludeMetaFilesFilter());
    
    // use the hint and max feasible parallelism to find an optimum degree for
    // parallelism
    numberOfJobs = Math.min(files.length, numberOfJobs);
    int filePerJob = (int) (Math.ceil(files.length / (float) numberOfJobs));
    numberOfJobs = (int) (Math.ceil(files.length / (float) filePerJob));
    
    //run the jobs
    Job[] jobs = new Job[numberOfJobs];
    int nextFileIndex = 0;
    for (int jobIndex = 0; jobIndex < numberOfJobs; jobIndex++) {
      SimpleColPartitionedTransposeJob job = new SimpleColPartitionedTransposeJob();
      Path subJobOutPath = new Path(matrixOutputPath, "" + jobIndex);
      List<Path> inFilesList = new ArrayList<Path>(filePerJob);
      int lastFileIndex = Math.min(nextFileIndex + filePerJob, files.length);
      for (; nextFileIndex < lastFileIndex; nextFileIndex++) {
        FileStatus fileStatus = files[nextFileIndex];
        if (fileStatus.isDir())//is {@link MapDir}
          inFilesList.add(new Path(fileStatus.getPath(), "data"));
        else
          inFilesList.add(fileStatus.getPath()); 
      }
      Path[] inFiles = new Path[inFilesList.size()];
      inFilesList.toArray(inFiles);
      jobs[jobIndex] = job.run(conf, inFiles, subJobOutPath, numInputRows,
          numInputCols, numColPartitions);
      boolean res = jobs[jobIndex].waitForCompletion(true);
      if (!res)
        throw new IOException("Job failed! " + jobIndex);
    }
    
    //wait for the jobs (in case they are run in parallel and move their output 
    //to the main output directory
    for (int jobIndex = 0; jobIndex < numberOfJobs; jobIndex++) {
      boolean res = jobs[jobIndex].waitForCompletion(true);
      if (!res)
        throw new IOException("Job failed! " + jobIndex);
//      Path subJobDir = new Path(matrixOutputPath, "" + jobIndex);
//      FileStatus[] jobOutFiles = fs.listStatus(subJobDir, new ExcludeMetaFilesFilter());
//      for (FileStatus jobOutFile : jobOutFiles) {
//        Path src = jobOutFile.getPath();
//        String name = src.getName();
//        name = name.replace("--", "-");//to circumvent the bug that inserts an extra "-"
//        Path dst = new Path(matrixOutputPath, name + "-j-" + jobIndex);
//        //unique name by indexing with folder id
//        log.info("fs.rename " + src + " -> " + dst);
//        fs.rename(src, dst);
//      }
//      log.info("fs.delete " + subJobDir);
//      fs.delete(subJobDir, true);
    }
  }

  /**
   * Perform transpose of A, where A refers to the path that contain a matrix
   * in {@link SequenceFileInputFormat}. Refer to
   * {@link SimpleColPartitionedTransposeJob} for further details.
   * 
   * @param conf
   *          the initial configuration
   * @param matrixInputPaths
   *          the list of paths to the input files that we process
   * 
   * @param matrixOutputPath
   *          the path of the resulting transpose matrix
   * @param numInputRows
   *          rows
   * @param numInputCols
   *          cols
   * @param numColPartitions
   *          the hint for the desired number of column partitions
   * @return the running job
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public Job run(Configuration conf, Path matrixInputPaths[],
      Path matrixOutputPath, int numInputRows, int numInputCols, int numColPartitions) 
          throws IOException, InterruptedException, ClassNotFoundException {
    conf.setInt(NUM_ORIG_ROWS_KEY, numInputRows);
    conf.setInt(NUM_COL_PARTITIONS, numColPartitions);
    conf.setInt(RowPartitioner.TOTAL_KEYS, numInputCols);
    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(SimpleColPartitionedTransposeJob.class);
    job.setJobName(SimpleColPartitionedTransposeJob.class.getSimpleName());

    FileSystem fs = FileSystem.get(matrixOutputPath.toUri(), conf);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    for (Path path : matrixInputPaths) {
      path = fs.makeQualified(path);
      MultipleInputs.addInputPath(job, path, SequenceFileInputFormat.class);
    }
    
    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setMapperClass(TransposeMapper.class);
    job.setMapOutputKeyClass(ColPartitionedRowWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    job.setPartitionerClass(IntSplittedRowPartitioner.class);
//    int numReducers = NMFCommon.getNumberOfReduceSlots(conf, "transpose-cp");
//    job.setNumReduceTasks(numReducers / numColPartitions);
  job.setNumReduceTasks(1);

    
    job.setCombinerClass(MergeVectorsCombiner.class);
    job.setReducerClass(MergeVectorsReducer.class);
    // one reducer means that the aggregate map output of the future
    // multiplication jobs will be smaller due to better efficiency of combiners
//    job.setNumReduceTasks(1);
//    job.setOutputFormatClass(LazyOutputFormat.class);
//    LazyOutputFormat.setOutputFormatClass(job, SequenceFileOutputFormat.class);
    job.setOutputFormatClass(MatrixOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.submit();
    return job;
  }

  public static class IntSplittedRowPartitioner<V> extends
      RowPartitioner<ColPartitionedRowWritable, V> {
    @Override
    public int getPartition(ColPartitionedRowWritable key, V value,
        int numPartitions) {
      return getPartition(key.getRowId(), numPartitions);
    }
  }

  /**
   * A row id augmented with column partition id, used as the output of mappers
   * @author myabandeh
   *
   */
  public static class ColPartitionedRowWritable extends WritableComparator
      implements WritableComparable<ColPartitionedRowWritable> {
    private short colPartitionId;
    private int rowId;

    public ColPartitionedRowWritable() {
      super(ColPartitionedRowWritable.class);
    }

    public short getColPartitionId() {
      return colPartitionId;
    }

    public int getRowId() {
      return rowId;
    }

    public ColPartitionedRowWritable(int r, int c, int totalCols, short totalPartitions) {
      this();
      set(r, c, totalCols, totalPartitions);
    }

    public void set(int r, int c, int totalCols, short totalPartitions) {
      rowId = r;
      colPartitionId = (short) ((c / (float) totalCols) * totalPartitions);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
      rowId = in.readInt();
      colPartitionId = in.readShort();
    }

    @Override
    public void write(DataOutput out) throws IOException {
      out.writeInt(rowId);
      out.writeShort(colPartitionId);
    }

    /**
     * Not only the output should be sorted by row id but also the rows with the same 
     * column id should be together, otherwise the reducer generates many files
     * for each column partition
     */
    @Override
    public int compareTo(ColPartitionedRowWritable o) {
      return ComparisonChain.start().compare(rowId, o.rowId).
          compare(colPartitionId, o.colPartitionId).result();
    }

    @Override
    public int hashCode() {
      return new HashCodeBuilder(17, 37).append(rowId).append(colPartitionId)
          .toHashCode();
    }

    @Override
    public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
      int r1 = readInt(b1, s1);
      int r2 = readInt(b2, s2);
      if (r1 != r2)
        return r1 - r2;
      s1 += 4;
      s2 += 4;
      short cpid1 = (short)readUnsignedShort(b1, s1);
      short cpid2 = (short)readUnsignedShort(b2, s2);
      return cpid1 - cpid2;
    }

    @Override
    public String toString() {
      return "ColPartitionedRow[" + rowId + "]-" + colPartitionId;
    }
  }
  
  /**
   * Send each element tagged with the column partition to reducers
   * @author myabandeh
   *
   */
  public static class TransposeMapper extends
      Mapper<IntWritable, VectorWritable, ColPartitionedRowWritable, VectorWritable> {
    private int newNumCols;
    private short numColPartitions;
    private ColPartitionedRowWritable cprw = new ColPartitionedRowWritable();

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      newNumCols = conf.getInt(NUM_ORIG_ROWS_KEY, Integer.MAX_VALUE);
      int numColPartitionsInt = conf.getInt(NUM_COL_PARTITIONS, 1);
      if (numColPartitionsInt >= Short.MAX_VALUE)
        throw new IllegalArgumentException("Too many column partitions: "
            + numColPartitionsInt);
      numColPartitions = (short) numColPartitionsInt;
    }

    @Override
    public void map(IntWritable r, VectorWritable v, Context context)
        throws IOException, InterruptedException {
      int row = r.get();
      Iterator<Vector.Element> it = v.get().nonZeroes().iterator();
      while (it.hasNext()) {
        Vector.Element e = it.next();
        RandomAccessSparseVector tmp = new RandomAccessSparseVector(newNumCols,
            1);
        tmp.setQuick(row, e.get());
        cprw.set(e.index(), row, newNumCols, numColPartitions);
        context.write(cprw, new VectorWritable(tmp));
      }
    }
    
  }

  /**
   * Merge the 1-element vectors generated by the mapper to a bigger vector
   * @author myabandeh
   *
   */
  public static class MergeVectorsCombiner
      extends
      Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {

    @Override
    public void reduce(WritableComparable<?> key,
        Iterable<VectorWritable> vectors, Context context) throws IOException,
        InterruptedException {
      context.write(key, VectorWritable.merge(vectors.iterator()));
    }
  }

  /**
   * Merge vectors generated by mappers, but write them to different files based on 
   * the column partition id
   * @author myabandeh
   *
   */
  public static class MergeVectorsReducer
      extends
      Reducer<ColPartitionedRowWritable, VectorWritable, IntWritable, VectorWritable> {
    private MultipleOutputs<IntWritable, VectorWritable> out;
    private IntWritable iw = new IntWritable();
    
    @Override
    public void setup(Context context) {
      out = new MultipleOutputs<IntWritable, VectorWritable>(context);
    }
    
    @Override
    public void reduce(ColPartitionedRowWritable key,
        Iterable<VectorWritable> vectors, Context context) throws IOException,
        InterruptedException {
      Vector merged = VectorWritable.merge(vectors.iterator()).get();
      iw.set(key.getRowId());
      String outName = getOutName(key.getColPartitionId());
      out.write(iw,
          new VectorWritable(new SequentialAccessSparseVector(merged)), outName);
    }
    
    private String getOutName(short colPartitionId) {
      return "part" + "-" + "cp" + "-" + colPartitionId;
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      out.close();
    }
  }
}
