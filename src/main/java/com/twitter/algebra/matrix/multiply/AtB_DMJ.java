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
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.twitter.algebra.matrix.format.MapDir;
import com.twitter.algebra.matrix.format.MatrixOutputFormat;
import com.twitter.algebra.matrix.format.RowPartitioner;
import com.twitter.algebra.nmf.ColPartitionJob;
import com.twitter.algebra.nmf.NMFCommon;

/**
 * Perform A x B matrix multiplication
 * 
 * Approach: Outer-join
 * 
 * Number of jobs: 1
 * 
 * Assumption: (1) Transpose At is already available, (2) the smaller of At and
 * B is in {@link MapDir} format, (3) the bigger of At and B is in partially
 * sorted sequence file format--Partial sort meaning that the input to each
 * mapper is sorted.
 * 
 * Design: Iterate over the rows of the bigger of At and B. For each row, load
 * the corresponding row from the other matrix, which is efficient since the
 * other is in {@link MapDir} format and the input rows are partially sorted.
 * Each mapper perform Ati x Bi (row i of At and row i of B) multiplication and
 * generates partial matrix Ci. The reducers sum up partial Ci matrices to get C
 * = A x B.
 * 
 * We can also partition the larger of A and B based on columns. This allows
 * denser partial C matrices and hence much less load on the combiners. If the
 * partial C matrix is small enough we can even use an in-memory combiner.
 */
public class AtB_DMJ extends AbstractJob {
  private static final Logger log = LoggerFactory.getLogger(AtB_DMJ.class);

  public static final String MATRIXINMEMORY = "matrixInMemory";
  /**
   * The smaller of the matrices is chosen to be loaded at runtime via its
   * {@link MapDir} format. This config parameter specifies which will be loaded
   * as {@link MapDir}.
   */
  public static final String AISMAPDIR = "matrix.a.is.mapdir";
  public static final String RESULTROWS = "matrix.result.num.rows";
  public static final String RESULTCOLS = "matrix.result.num.cols";
  public static final String PARTITIONCOLS = "matrix.partition.num.cols";
  public static final String USEINMEMCOMBINER = "matrix.useInMemCombiner";
  
  /**
   * @param args
   * @throws Exception
   */
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new AtB_DMJ(), args);
  }
  
  @Override
  public int run(String[] strings) throws Exception {
    addOutputOption();
    addOption("numColsAt", "nca",
        "Number of columns of the first input matrix", true);
    addOption("numColsB", "ncb",
        "Number of columns of the second input matrix", true);
    addOption("atMatrix", "atMatrix", "The first matrix, transposed");
    addOption("bMatrix", "bMatrix", "The second matrix");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    Path atPath = new Path(getOption("atMatrix"));
    Path bPath = new Path(getOption("bMatrix"));
    int atCols = Integer.parseInt(getOption("numColsAt"));
    int bCols = Integer.parseInt(getOption("numColsB"));
    run(getConf(), atPath, bPath, getOutputPath(), atCols, bCols, 1, 1, true);
    return 0;
  }

  /**
   * Perform A x B, where At and B are already wrapped in a DistributedRowMatrix
   * object. Refer to {@link AtB_DMJ} for further details.
   * 
   * Automatically decide on partitioning the larger matrix to be used with
   * in-memory combiners.
   * 
   * @param conf the initial configuration
   * @param At transpose of matrix A
   * @param B matrix B
   * @param label the label for the output directory
   * @param labelAtCol by using a fixed label for AtCol one can avoid the second
   *          run of the partitioning job if we know that At is not changed
   * @param lableBCol by using a fixed label for BCol one can avoid the second
   *          run of the partitioning job if we know that B is not changed
   * @return AxB wrapped in a DistributedRowMatrix object
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public static DistributedRowMatrix smartRun(Configuration conf,
      DistributedRowMatrix At, DistributedRowMatrix B, String label, String labelAtCol, String lableBCol) 
          throws IOException,
      InterruptedException, ClassNotFoundException {
    log.info("running " + AtB_DMJ.class.getName());
    if (At.numRows() != B.numRows())
      throw new CardinalityException(At.numRows(), B.numRows());
    Path outPath = new Path(At.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    AtB_DMJ job = new AtB_DMJ();
    if (!fs.exists(outPath)) {
      int numColPartitionsAt = 1, numColPartitionsB = 1;
      int numColPartitions = NMFCommon.computeOptColPartitionsForMemCombiner(conf, At.numCols(), B.numCols());
      long atSize = MapDir.du(At.getRowPath(), fs);
      long bSize = MapDir.du(B.getRowPath(), fs);
      //cost is size of remote reads. For each col partition we need to read the entire of the other matrix once
      long atPartitionCost = numColPartitions * bSize;
      long bPartitionCost = numColPartitions * atSize;
      log.info("smart partitioning: numColPartitions: " + numColPartitions
          + " atSize: " + atSize + " bSize: " + bSize + " atCost="
          + atPartitionCost + " vs.  bCost=" + bPartitionCost);
      if (atPartitionCost < bPartitionCost) {
        At =  ColPartitionJob.partition(At, conf, labelAtCol, numColPartitions);
        numColPartitionsAt = numColPartitions;
      } else {
        B =  ColPartitionJob.partition(B, conf, lableBCol, numColPartitions);
        numColPartitionsB = numColPartitions;
      }
      job.run(conf, At.getRowPath(), B.getRowPath(), outPath, At.numCols(),
          B.numCols(), numColPartitionsAt,  numColPartitionsB, true);
    } else {
      log.warn("----------- Skip already exists: " + outPath);
    }
    DistributedRowMatrix distRes =
        new DistributedRowMatrix(outPath, At.getOutputTempPath(), At.numCols(),
            B.numCols());
    distRes.setConf(conf);
    return distRes;
  }
  
  /**
   * Perform A x B, where At and B are already wrapped in a DistributedRowMatrix
   * object. Refer to {@link AtB_DMJ} for further details.
   * 
   * @param conf the initial configuration
   * @param At transpose of matrix A
   * @param B matrix B
   * @param numColPartitionsAt 
   * @param numColPartitionsB 
   * @param label the label for the output directory
   * @param useInMemCombiner
   * @return AxB wrapped in a DistributedRowMatrix object
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public static DistributedRowMatrix run(Configuration conf,
      DistributedRowMatrix At, DistributedRowMatrix B, int numColPartitionsAt,
      int numColPartitionsB, String label, boolean useInMemCombiner) throws IOException,
      InterruptedException, ClassNotFoundException {
    log.info("running " + AtB_DMJ.class.getName());
    if (At.numRows() != B.numRows())
      throw new CardinalityException(At.numRows(), B.numRows());
    if (numColPartitionsAt != 1 && numColPartitionsB != 1)
      throw new IOException("AtB_DMJ: not both At and B can be column partitioned!");
    Path outPath = new Path(At.getOutputTempPath(), label);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    AtB_DMJ job = new AtB_DMJ();
    if (!fs.exists(outPath)) {
      job.run(conf, At.getRowPath(), B.getRowPath(), outPath, At.numCols(),
          B.numCols(), numColPartitionsAt,  numColPartitionsB, useInMemCombiner);
    } else {
      log.warn("----------- Skip already exists: " + outPath);
    }
    DistributedRowMatrix distRes =
        new DistributedRowMatrix(outPath, At.getOutputTempPath(), At.numCols(),
            B.numCols());
    distRes.setConf(conf);
    return distRes;
  }

  /**
   * Perform A x B, where A and B refer to the paths that contain matrices in
   * {@link SequenceFileInputFormat}. The smaller of At and B must also conform
   * with {@link MapDir} format. Refer to {@link AtB_DMJ} for further details.
   * 
   * @param conf the initial configuration
   * @param atPath path to transpose of matrix A.
   * @param bPath path to matrix B
   * @param matrixOutputPath path to which AxB will be written
   * @param atCols number of columns of At (rows of A)
   * @param bCols
   * @param numColPartitionsAt
   * @param numColPartitionsB 
   * @param useInMemCombiner
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public void run(Configuration conf, Path atPath, Path bPath,
      Path matrixOutputPath, int atCols, int bCols, int numColPartitionsAt,
      int numColPartitionsB, boolean useInMemCombiner) throws IOException, InterruptedException,
      ClassNotFoundException {
    boolean aIsMapDir = true;
    if (1 == numColPartitionsAt && 1 == numColPartitionsB) {// if we do not use col partitioning
      FileSystem fs = FileSystem.get(atPath.toUri(), conf);
      long atSize = MapDir.du(atPath, fs);
      long bSize = MapDir.du(bPath, fs);
      log.info("Choosing the smaller matrix: atSize: " + atSize + " bSize: "
          + bSize);
      aIsMapDir = atSize < bSize;
    } else if (numColPartitionsAt != 1) {
      aIsMapDir = false;
    } else if (numColPartitionsB != 1) {
      aIsMapDir = true;
    }
    AtB_DMJ job = new AtB_DMJ();
    Job hjob;
    if (aIsMapDir) {
      int colsPerPartition =
          ColPartitionJob.getColPartitionSize(bCols, numColPartitionsB);
      hjob =
          job.run(conf, atPath, bPath, matrixOutputPath, atCols, bCols,
              colsPerPartition, aIsMapDir, useInMemCombiner);
    } else {
      int colsPerPartition =
          ColPartitionJob.getColPartitionSize(atCols, numColPartitionsAt);
      hjob =
          job.run(conf, bPath, atPath, matrixOutputPath, atCols, bCols,
              colsPerPartition, aIsMapDir, useInMemCombiner);
    }
    boolean res = hjob.waitForCompletion(true);
    if (!res)
      throw new IOException("Job failed! ");
  }

  /**
   * Perform A x B, where At and B refer to the paths that contain matrices in
   * {@link SequenceFileInputFormat}. One of At and B must also conform with
   * {@link MapDir} format. Refer to {@link AtB_DMJ} for further details.
   * 
   * @param conf the initial configuration
   * @param mapDirPath path to the matrix in {@link MapDir} format
   * @param matrixInputPaths the list of paths to matrix input partitions over
   *          which we iterate
   * @param matrixOutputPath path to which AxB will be written
   * @param atCols number of columns of At (rows of A)
   * @param bCols
   * @param colsPerPartition cols per partition of the input matrix (whether At or B)
   * @param aIsMapDir is A chosen to be loaded as MapDir
   * @param useInMemCombiner
   * @param numberOfJobs the hint for the desired number of parallel jobs
   * @return the running job
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public Job run(Configuration conf, Path mapDirPath, Path matrixInputPaths,
      Path matrixOutputPath, int atCols, int bCols, int colsPerPartition,
      boolean aIsMapDir, boolean useInMemCombiner) throws IOException,
      InterruptedException, ClassNotFoundException {
    conf = new Configuration(conf);

    conf.set(MATRIXINMEMORY, mapDirPath.toString());
    conf.setBoolean(AISMAPDIR, aIsMapDir);
    conf.setBoolean(USEINMEMCOMBINER, useInMemCombiner);
    conf.setInt(RESULTROWS, atCols);
    conf.setInt(RESULTCOLS, bCols);
    conf.setInt(PARTITIONCOLS, colsPerPartition);
    FileSystem fs = FileSystem.get(matrixOutputPath.toUri(), conf);
    NMFCommon.setNumberOfMapSlots(conf, fs, matrixInputPaths, "dmj");

    if (useInMemCombiner) {
      Configuration newConf = new Configuration(conf);
      newConf.set("mapreduce.task.io.sort.mb", "1");
      conf = newConf;
    }

    @SuppressWarnings("deprecation")
    Job job = new Job(conf);
    job.setJarByClass(AtB_DMJ.class);
    job.setJobName(AtB_DMJ.class.getSimpleName());
    matrixOutputPath = fs.makeQualified(matrixOutputPath);

    matrixInputPaths = fs.makeQualified(matrixInputPaths);
    MultipleInputs.addInputPath(job, matrixInputPaths,
        SequenceFileInputFormat.class);

    FileOutputFormat.setOutputPath(job, matrixOutputPath);
    job.setMapperClass(MyMapper.class);
    job.setMapOutputKeyClass(IntWritable.class);
    job.setMapOutputValueClass(VectorWritable.class);

    if (!useInMemCombiner)
      job.setCombinerClass(AtBOuterStaticMapsideJoinJob.MyReducer.class);

    int numReducers = NMFCommon.getNumberOfReduceSlots(conf, "dmj");
    job.setNumReduceTasks(numReducers);
    // ensures total order (when used with {@link MatrixOutputFormat}),
    RowPartitioner.setPartitioner(job, RowPartitioner.IntRowPartitioner.class,
        atCols);

    job.setReducerClass(EpsilonReducer.class);
    job.setOutputFormatClass(MatrixOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.submit();
    return job;
  }

  /**
   * Iterate over the input vectors, do an outer join with the corresponding
   * vector from the other matrix, and write the resulting partial matrix to the
   * reducers.
   * 
   * @author myabandeh
   * 
   */
  public static class MyMapper extends
  Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    private MapDir otherMapDir;
    private boolean aIsMapDir;
    InMemCombiner inMemCombiner;
    private VectorWritable otherVectorw = new VectorWritable();
    private VectorWritable outVectorw = new VectorWritable();
    private IntWritable outIntw = new IntWritable();
    int resultRows, resultCols;
    int colsPerPartition;

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      Path inMemMatrixPath = new Path(conf.get(MATRIXINMEMORY));
      resultRows = conf.getInt(RESULTROWS, Integer.MAX_VALUE);
      resultCols = conf.getInt(RESULTCOLS, Integer.MAX_VALUE);
      colsPerPartition = conf.getInt(PARTITIONCOLS, Integer.MAX_VALUE);
      otherMapDir = new MapDir(conf, inMemMatrixPath);
      aIsMapDir = conf.getBoolean(AISMAPDIR, true);
      boolean useInMemCombiner = conf.getBoolean(USEINMEMCOMBINER, true);
      if (useInMemCombiner) {
        try {
          if (aIsMapDir) {
            inMemCombiner = new InMemCombinerColPartitionB();
            inMemCombiner.init(resultRows, colsPerPartition);
          } else {
            inMemCombiner = new InMemCombinerColPartitionAt();
            inMemCombiner.init(colsPerPartition, resultCols);
          }
        } catch (Exception e) {
          inMemCombiner = null;
          System.gc();
          System.err.println("Not enough mem for in memory combiner of size "
              + resultRows * colsPerPartition);
        }
      }
    }

    @Override
    public void map(IntWritable index, VectorWritable bvw, Context context)
        throws IOException, InterruptedException {
      Vector normalInput = bvw.get();
      Writable res = otherMapDir.get(index, otherVectorw);
      if (res == null) {
        // too many nulls could indicate a bug, good to check
        context.getCounter("MapDir", "nullValues").increment(1);
        return;
      }
      Vector mapDirVector = otherVectorw.get();
      if (aIsMapDir)
        outerProduct(mapDirVector, normalInput, context);
      else
        outerProduct(normalInput, mapDirVector, context);
    }

    void outerProduct(Vector aVector, Vector bVector, Context context)
        throws IOException, InterruptedException {
      Iterator<Vector.Element> it = aVector.nonZeroes().iterator();
      while (it.hasNext()) {
        Vector.Element e = it.next();
        if (inMemCombiner != null)
          inMemCombiner.combine(e.index(), e.get(), bVector);
        else {
          outIntw.set(e.index());
          outVectorw.set(bVector.times(e.get()));
          context.write(outIntw, outVectorw);
        }
      }
    }
    
    @Override
    public void cleanup(Context context) throws IOException {
      otherMapDir.close();
      if (inMemCombiner != null)
        inMemCombiner.dump(context);
    }

    interface InMemCombiner {
      void init(int rows, int cols);
      void combine(int row, double scale, Vector bVector);
      void dump(Context context) throws IOException;
    }
    
    
    class InMemCombinerColPartitionB implements InMemCombiner {
      private double[] outValues;
      @Override
      public void init(int rows, int cols) {
        outValues = new double[rows * cols];
      }
      
      @Override
      public void combine(int row, double scale, Vector bVector) {
        Iterator<Vector.Element> it = bVector.nonZeroes().iterator();
        while (it.hasNext()) {
          Vector.Element e = it.next();
          int localCol = globalToLocalCol(e.index());
          int localIndex = rowcolToIndex(row, localCol);
          outValues[localIndex] += scale * e.get();
        }
      }
      
      @Override
      public void dump(Context context) throws IOException {
        if (minGlobalCol == Integer.MIN_VALUE) {
          //too many null values could indicate an error
          context.getCounter("InMemCombiner", "nullValues").increment(1);
          return;
//          throw new IOException("minGlobalCol is not set!");
        }
        IntWritable iw = new IntWritable();
        VectorWritable vw = new VectorWritable();
        RandomAccessSparseVector vector =
            new RandomAccessSparseVector(resultCols, colsPerPartition);
        for (int r = 0; r < resultRows; r++) {
          int index = rowcolToIndex(r, 0);
          boolean hasNonZero = false;
          for (int c = 0; c < colsPerPartition; c++, index++) {
            double value = outValues[index];
            if (value != 0) {
              hasNonZero = true;
              vector.set(c + minGlobalCol, value);
            }
          }
          if (!hasNonZero)
            continue;
          iw.set(r);
          vw.set(new SequentialAccessSparseVector(vector));
          try {
            context.write(iw, vw);
          } catch (InterruptedException e) {
            context.getCounter("Error", "interrupt").increment(1);
          }
          vector = new RandomAccessSparseVector(resultCols, colsPerPartition);
        }
      }

      private int minGlobalCol = Integer.MIN_VALUE;

      private int globalToLocalCol(int col) {
        int localCol = col % colsPerPartition;
        if (minGlobalCol == Integer.MIN_VALUE)
          minGlobalCol = col - localCol;
        return localCol;
      }

      private int rowcolToIndex(int row, int localCol) {
        return row * colsPerPartition + localCol;
      }
    }

    class InMemCombinerColPartitionAt implements InMemCombiner {
      private double[] outValues;
      @Override
      public void init(int rows, int cols) {
        outValues = new double[rows * cols];
      }
      
      @Override
      public void combine(int row, double scale, Vector bVector) {
        Iterator<Vector.Element> it = bVector.nonZeroes().iterator();
        while (it.hasNext()) {
          Vector.Element e = it.next();
          int localRow = globalToLocalRow(row);
          int localIndex = rowcolToIndex(localRow, e.index());
          outValues[localIndex] += scale * e.get();
        }
      }
      
      @Override
      public void dump(Context context) throws IOException {
        if (minGlobalRow == Integer.MIN_VALUE) {
          //too many null values could indicate an error
          context.getCounter("InMemCombiner", "nullValues").increment(1);
          return;
        }
        IntWritable iw = new IntWritable();
        VectorWritable vw = new VectorWritable();
        RandomAccessSparseVector vector =
            new RandomAccessSparseVector(resultCols, resultCols);
        for (int r = 0; r < colsPerPartition; r++) {
          int index = rowcolToIndex(r, 0);
          boolean hasNonZero = false;
          for (int c = 0; c < resultCols; c++, index++) {
            double value = outValues[index];
            if (value != 0) {
              hasNonZero = true;
              vector.set(c, value);
            }
          }
          if (!hasNonZero)
            continue;
          iw.set(r + minGlobalRow);
          vw.set(new SequentialAccessSparseVector(vector));
          try {
            context.write(iw, vw);
          } catch (InterruptedException e) {
            context.getCounter("Error", "interrupt").increment(1);
          }
          vector = new RandomAccessSparseVector(resultCols, resultCols);
        }
      }

      private int minGlobalRow = Integer.MIN_VALUE;

      private int globalToLocalRow(int row) {
        int localRow = row % colsPerPartition;
        if (minGlobalRow == Integer.MIN_VALUE)
          minGlobalRow = row - localRow;
        return localRow;
      }

      private int rowcolToIndex(int row, int localCol) {
        return row * resultCols + localCol;
      }
    }

  }


  public static class EpsilonReducer extends
      Reducer<IntWritable,VectorWritable,IntWritable,VectorWritable> {
    static double EPSILON = Double.NaN;
    static final String EPSILON_STR = "matrix.atb.epsilon";
    private VectorWritable outvw = new VectorWritable();
    
    @Override
    public void setup(Context context) throws IOException {
      EPSILON = context.getConfiguration().getFloat(EPSILON_STR, Float.NaN);
    }

    @Override
    public void reduce(IntWritable rowNum, Iterable<VectorWritable> values,
        Context context) throws IOException, InterruptedException {
      Iterator<VectorWritable> it = values.iterator();
      if (!it.hasNext())
        return;
      RandomAccessSparseVector accumulator =
          new RandomAccessSparseVector(it.next().get());
      while (it.hasNext()) {
        Vector row = it.next().get();
        accumulator.assign(row, Functions.PLUS);
      }
      accumulator = zeroize(accumulator);
      Vector resVector = new SequentialAccessSparseVector(accumulator);
      outvw.set(resVector);
      context.write(rowNum, outvw);
    }

    RandomAccessSparseVector zeroize(RandomAccessSparseVector vector) {
      if (Double.isNaN(EPSILON))
        return vector;
      return new FilterEpsilonRandomAccessSparseVector(vector);
    }
    
    class FilterEpsilonRandomAccessSparseVector extends RandomAccessSparseVector {
      FilterEpsilonRandomAccessSparseVector(RandomAccessSparseVector unfiltered) {
        super(unfiltered.size(), unfiltered.getNumNondefaultElements());
        for (Element e : unfiltered.nonZeroes()) {
          if (e.get() > EPSILON)
            setQuick(e.index(), e.get());
        }
      }
    }
  }
}
