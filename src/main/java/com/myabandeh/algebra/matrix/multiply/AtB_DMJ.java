package com.myabandeh.algebra.matrix.multiply;

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
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.CardinalityException;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.myabandeh.algebra.matrix.format.MapDir;
import com.myabandeh.algebra.matrix.format.MatrixOutputFormat;
import com.myabandeh.algebra.matrix.format.RowPartitioner;
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

    Path aPath = new Path(getOption("atMatrix"));
    Path bPath = new Path(getOption("bMatrix"));
    int atCols = Integer.parseInt(getOption("numColsAt"));
    int bCols = Integer.parseInt(getOption("numColsB"));
    run(getConf(), aPath, bPath, getOutputPath(), atCols, bCols, 1, 1, true);
    return 0;
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
    if (1 == numColPartitionsB) {// if we do not use col partitioning on B
      FileSystem fs = FileSystem.get(atPath.toUri(), conf);
      long atSize = MapDir.du(atPath, fs);
      long bSize = MapDir.du(bPath, fs);
      log.info("Choosing the smaller matrix: atSize: " + atSize + " bSize: "
          + bSize);
      aIsMapDir = atSize < bSize;
    }
    int colsPerPartition =
        ColPartitionJob.getColPartitionSize(bCols, numColPartitionsB);
    AtB_DMJ job = new AtB_DMJ();
    Job hjob;
    if (aIsMapDir)
      hjob =
          job.run(conf, atPath, bPath, matrixOutputPath, atCols, bCols,
              colsPerPartition, aIsMapDir, useInMemCombiner);
    else
      hjob =
          job.run(conf, bPath, atPath, matrixOutputPath, atCols, bCols,
              colsPerPartition, aIsMapDir, useInMemCombiner);
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
   * @param colsPerPartition
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
    conf.set(MATRIXINMEMORY, mapDirPath.toString());
    conf.setBoolean(AISMAPDIR, aIsMapDir);
    conf.setBoolean(USEINMEMCOMBINER, useInMemCombiner);
    conf.setInt(RESULTROWS, atCols);
    conf.setInt(RESULTCOLS, bCols);
    conf.setInt(PARTITIONCOLS, colsPerPartition);
    FileSystem fs = FileSystem.get(matrixOutputPath.toUri(), conf);
    NMFCommon.setNumberOfMapSlots(conf, fs, matrixInputPaths, "dmj");

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

    job.setReducerClass(AtBOuterStaticMapsideJoinJob.MyReducer.class);
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
    int partitionCols;

    @Override
    public void setup(Context context) throws IOException {
      Configuration conf = context.getConfiguration();
      Path inMemMatrixPath = new Path(conf.get(MATRIXINMEMORY));
      resultRows = conf.getInt(RESULTROWS, Integer.MAX_VALUE);
      resultCols = conf.getInt(RESULTCOLS, Integer.MAX_VALUE);
      partitionCols = conf.getInt(PARTITIONCOLS, Integer.MAX_VALUE);
      otherMapDir = new MapDir(conf, inMemMatrixPath);
      aIsMapDir = conf.getBoolean(AISMAPDIR, true);
      boolean useInMemCombiner = conf.getBoolean(USEINMEMCOMBINER, true);
      if (useInMemCombiner) {
        try {
          inMemCombiner = new InMemCombinerColPartitionB();
          inMemCombiner.init(resultRows, partitionCols);
        } catch (Exception e) {
          inMemCombiner = null;
          System.gc();
          useInMemCombiner = false;
          System.err.println("Not enough mem for in memory combiner of size "
              + resultRows * partitionCols);
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
            new RandomAccessSparseVector(resultCols, partitionCols);
        for (int r = 0; r < resultRows; r++) {
          int index = rowcolToIndex(r, 0);
          boolean hasNonZero = false;
          for (int c = 0; c < partitionCols; c++, index++) {
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
            context.getCounter("Error", "interrup").increment(1);
          }
          vector = new RandomAccessSparseVector(resultCols, partitionCols);
        }
      }

      private int minGlobalCol = Integer.MIN_VALUE;

      private int globalToLocalCol(int col) {
        int localCol = col % partitionCols;
        if (minGlobalCol == Integer.MIN_VALUE)
          minGlobalCol = col - localCol;
        return localCol;
      }

      private int rowcolToIndex(int row, int localCol) {
        return row * partitionCols + localCol;
      }
    }

  }

}
