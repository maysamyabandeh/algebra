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

package com.twitter.algebra;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapFile;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.Closeables;
import com.twitter.algebra.matrix.format.MapDir;
import com.twitter.algebra.matrix.format.MatrixOutputFormat;

/**
 * This class includes the utility functions that is used by multiple algebra classes
 * 
 * @author maysam yabandeh
 */
public class AlgebraCommon {
  public static class ExcludeMetaFilesFilter implements PathFilter {
    @Override
    public boolean accept(Path path) {
      String name = path.getName();
      return !name.startsWith(".") && !name.startsWith("_");
    }
  }

  private static final Logger log = LoggerFactory
      .getLogger(AlgebraCommon.class);

  /**
   * @param m
   *          matrix
   * @return m.viewDiagonal().zSum()
   */
  static double trace(Matrix m) {
    Vector d = m.viewDiagonal();
    return d.zSum();
  }

  /**
   * Assuming that the input is generated by {@link MatrixOutputFormat}, this method
   * convert it to a centralized dense matrix.
   * @param inPath the path to the {@link MapDir matrix}
   * @param nRows
   * @param nCols
   * @param conf
   * @return
   * @throws IOException
   */
  public static DenseMatrix mapDirToDenseMatrix(Path inPath, int nRows, int nCols,
      Configuration conf) throws IOException {
    Path tmpPath = inPath.getParent();
    DistributedRowMatrix distMatrix = new DistributedRowMatrix(inPath, tmpPath,
        nRows, nCols);
    distMatrix.setConf(conf);
    return toDenseMatrix(distMatrix);
  }
  
  /**
   * Assuming that the input is generated by {@link MatrixOutputFormat}, this method
   * convert its first row it to a centralized sparse matrix.
   * @param inPath the path to the {@link MapDir matrix}
   * @param nRows
   * @param nCols
   * @param conf
   * @return
   * @throws IOException
   */
  public static Vector mapDirToSparseVector(Path inPath, int nRows, int nCols,
      Configuration conf) throws IOException {
    SparseMatrix matrix = mapDirToSparseMatrix(inPath, nRows, nCols, conf);
    Vector v = matrix.viewRow(0);
    return v;
  }
  
  /**
   * Assuming that the input is generated by {@link MatrixOutputFormat}, this method
   * convert it to a centralized sparse matrix.
   * @param inPath the path to the {@link MapDir matrix}
   * @param nRows
   * @param nCols
   * @param conf
   * @return
   * @throws IOException
   */
  public static SparseMatrix mapDirToSparseMatrix(Path inPath, int nRows, int nCols,
      Configuration conf) throws IOException {
    Path tmpPath = inPath.getParent();
    DistributedRowMatrix distMatrix = new DistributedRowMatrix(inPath, tmpPath,
        nRows, nCols);
    distMatrix.setConf(conf);
    return toSparseMatrix(distMatrix);
  }

  /***
   * If the MapDir matrix is small, we can convert it to an in memory representation
   * and then run efficient centralized operations
   * 
   * @param origMtx in MapDir format (generated by MatrixOutputFormat)
   * @return a dense matrix including the data
   * @throws IOException 
   */
  public static DenseMatrix toDenseMatrix(DistributedRowMatrix origMtx) throws IOException {
    MapDir mapDir = new MapDir(new Configuration(), origMtx.getRowPath());
    DenseMatrix mtx = new DenseMatrix(origMtx.numRows(), origMtx.numCols());
    Iterator<MatrixSlice> sliceIterator;
    try {
      sliceIterator = mapDir.iterateAll();
    } catch (Exception e) {
      log.info(e.toString());
      log.info("Input is not in matrix format, trying SequenceFileFormat instead ...");
      sliceIterator = origMtx.iterateAll();
    }
    while (sliceIterator.hasNext()) {
      MatrixSlice slice = sliceIterator.next();
//      int r = slice.index();
//      for (int c = 0; c < mtx.numCols(); c++) {
//        mtx.set(r, c, slice.get(c));
//      }
      mtx.viewRow(slice.index()).assign(slice.vector());
    }
    mapDir.close();
    return mtx;
  }

  /***
   * If the MapDir matrix is small, we can convert it to an in memory representation
   * and then run efficient centralized operations
   * 
   * @param origMtx in MapDir format (generated by MatrixOutputFormat)
   * @return a dense matrix including the data
   * @throws IOException 
   */
  static SparseMatrix toSparseMatrix(DistributedRowMatrix origMtx) throws IOException {
    MapDir mapDir = new MapDir(new Configuration(), origMtx.getRowPath());
    SparseMatrix mtx = new SparseMatrix(origMtx.numRows(), origMtx.numCols());
    Iterator<MatrixSlice> sliceIterator = mapDir.iterateAll();
    while (sliceIterator.hasNext()) {
      MatrixSlice slice = sliceIterator.next();
      mtx.viewRow(slice.index()).assign(slice.vector());
    }
    mapDir.close();
    return mtx;
  }

  /**
   * Trace of a matrix obtained in a centralized way. For some reason, which I did not have time to debug, raises memory exception for big matrices. 
   * 
   * TODO: MapReduce job for traces of big matrices.
   * @param origMtx
   * @return trace of the input matrix
   * @throws IOException
   */
  static double trace(DistributedRowMatrix origMtx) throws IOException {
    MapDir mapDir = new MapDir(new Configuration(), origMtx.getRowPath());
    Iterator<MatrixSlice> sliceIterator = mapDir.iterateAll();
    double trace = 0;
    while (sliceIterator.hasNext()) {
      MatrixSlice slice = sliceIterator.next();
      int index = slice.index();
      if (index >= slice.vector().size())
        break;
      double value = slice.vector().get(index);
      trace += Double.isNaN(value) ? 0 : value;
    }
    mapDir.close();
    return trace;
  }

  /**
   * Multiply a vector with a matrix
   * @param vector V
   * @param matrix M
   * @param resVector will be filled with V * M
   * @return V * M
   */
  public static Vector vectorTimesMatrix(Vector vector, Matrix matrix,
      DenseVector resVector) {
    int nCols = matrix.numCols();
    for (int c = 0; c < nCols; c++) {
      Double resDouble = vector.dot(matrix.viewColumn(c));
      resVector.set(c, resDouble);
    }
    return resVector;
  }

  /**
   * Multiply a vector with transpose of a matrix
   * @param vector V
   * @param transpose of matrix M
   * @param resVector will be filled with V * M
   * @return V * M
   */
  public static Vector vectorTimesMatrixTranspose(Vector vector, Matrix matrixTranspose,
      Vector resVector) {
    int nCols = matrixTranspose.numRows();
    for (int c = 0; c < nCols; c++) {
      Vector col = matrixTranspose.viewRow(c);
      double resDouble = 0d;
      boolean hasNonZero = col.getNumNondefaultElements() != 0;
      if (hasNonZero)
        resDouble = vector.dot(col);
      resVector.set(c, resDouble);
    }
    return resVector;
  }

  /**
   * Convert a 2-dimensional array to a dense matrix in {@link MapDir} format
   * @param vectors a 2-dimensional array of doubles
   * @param outPath the path to which the dense matrix will be written
   * @param tmpPath an argument required to be passed to {@link DistributedRowMatrix}
   * @param label a unique label to name the output matrix directory
   * @return a {@link DistributedRowMatrix} pointing to the in-filesystem matrix
   * @throws Exception
   */
  public static DistributedRowMatrix toDenseMapDir(double[][] vectors,
      Path outPath, Path tmpPath, String label) throws Exception {
    DenseMatrix m = new DenseMatrix(vectors);
    return AlgebraCommon.toMapDir(m, outPath, tmpPath, label);
  }

  /**
   * Convert a 2-dimensional array to a sparse matrix in {@link MapDir} format
   * @param vectors a 2-dimensional array of doubles
   * @param outPath the path to which the dense matrix will be written
   * @param tmpPath an argument required to be passed to {@link DistributedRowMatrix}
   * @param label a unique label to name the output matrix directory
   * @return a {@link DistributedRowMatrix} pointing to the in-filesystem matrix
   * @throws Exception
   */
  public static DistributedRowMatrix toSparseMapDir(double[][] vectors,
      Path outPath, Path tmpPath, String label) throws Exception {
    int nRows = vectors.length;
    int nCols = vectors[0].length;
    SparseMatrix m = new SparseMatrix(nRows, nCols);
    for (int r = 0; r < nRows; r++)
      m.set(r, vectors[r]);
    return AlgebraCommon.toMapDir(m, outPath, tmpPath, label);
  }

  /**
   * Convert an in-memory representation of a matrix to a distributed MapDir
   * format. It then can be used in distributed jobs
   * 
   * @param oriMatrix
   * @return path that will contain the matrix files
   * @throws Exception
   */
  public static DistributedRowMatrix toMapDir(Matrix origMatrix,
      Path outPath, Path tmpPath, String label) throws Exception {
    Configuration conf = new Configuration();
    Path outputDir = new Path(outPath, label + origMatrix.numRows() + "x"
        + origMatrix.numCols());
    FileSystem fs = FileSystem.get(outputDir.toUri(), conf);
    if (!fs.exists(outputDir)) {
      Path mapDir = new Path(outputDir, "matrix-k-0");
      Path outputFile = new Path(mapDir, "data");
      @SuppressWarnings("deprecation")
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf,
          outputFile, IntWritable.class, VectorWritable.class);
      VectorWritable vectorw = new VectorWritable();
      IntWritable intw = new IntWritable();
      try {
        for (int r = 0; r < origMatrix.numRows(); r++) {
          Vector vector = origMatrix.viewRow(r);
          vectorw.set(vector);
          intw.set(r);
          writer.append(intw, vectorw);
        }
      } finally {
        writer.close();
      }
      MapFile.fix(fs, mapDir, IntWritable.class, VectorWritable.class, false, conf);
    } else {
      log.warn("----------- Skip matrix " + outputDir + " - already exists");
    }
    DistributedRowMatrix dMatrix = new DistributedRowMatrix(outputDir, tmpPath,
        origMatrix.numRows(), origMatrix.numCols());
    dMatrix.setConf(conf);
    return dMatrix;
  }

  /**
   * Write a vector to filesystem so that it can be used by distributed jobs
   * @param vector
   * @param outputDir
   * @param label the unique label that be used in naming the vector file
   * @param conf
   * @return
   * @throws IOException
   */
  public static Path toDistributedVector(Vector vector, Path outputDir, String label,
      Configuration conf) throws IOException {
    Path outputFile = new Path(outputDir, "Vector-" + label);
    FileSystem fs = FileSystem.get(outputDir.toUri(), conf);
    if (fs.exists(outputFile)) {
      log.warn("----------- OVERWRITE " + outputFile + " already exists");
      fs.delete(outputFile, false);
    }
    @SuppressWarnings("deprecation")
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, outputFile,
        IntWritable.class, VectorWritable.class);
    VectorWritable vectorw = new VectorWritable();
    vectorw.set(vector);
    writer.append(new IntWritable(0), vectorw);
    writer.close();
    return outputFile;
  }

  /**
   * Read a vector from the filesystem and covert it to a dense vector
   * TODO: how about sparse vectors
   * @param vectorFile that file that contains the vector data in SequenceFile format
   * @param conf
   * @return a dense vector
   * @throws IOException
   */
  public static DenseVector toDenseVector(Path vectorFile, Configuration conf)
      throws IOException {
    SequenceFileIterator<IntWritable, VectorWritable> iterator = new SequenceFileIterator<IntWritable, VectorWritable>(
        vectorFile, true, conf);
    DenseVector vector;
    try {
      Pair<IntWritable, VectorWritable> next;
      next = iterator.next();
      vector = new DenseVector(next.getSecond().get());
    } finally {
      Closeables.close(iterator, false);
    }
    return vector;
  }

  static void printMemUsage() {
    int mb = 1024 * 1024;
    // Getting the runtime reference from system
    Runtime runtime = Runtime.getRuntime();
    System.out.println("##### Heap utilization statistics [MB] #####");
    // Print used memory
    System.out.print("Used Memory:"
        + (runtime.totalMemory() - runtime.freeMemory()) / mb);
    // Print free memory
    System.out.print(" Free Memory:" + runtime.freeMemory() / mb);
    // Print total available memory
    System.out.print(" Total Memory:" + runtime.totalMemory() / mb);
    // Print Maximum available memory
    System.out.print(" Max Memory:" + runtime.maxMemory() / mb);
  }

//  public static int computeNumReducers(Configuration conf, int rows, int cols) {
//    int blockSize = 100 * 1024 * 1024;
//    blockSize = conf.getInt("file.blocksize", blockSize);
//    int nReducers = (int) (rows / (float) blockSize * cols * 12);
//    // ~12 byte/element
//    nReducers = Math.max(nReducers, 1);
//    return nReducers;
//  }
}
