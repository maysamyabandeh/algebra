package com.twitter.algebra;

import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class MergeVectorsReducer
    extends
    Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {
  @Override
  public void reduce(WritableComparable<?> key,
      Iterable<VectorWritable> vectors, Context context) throws IOException,
      InterruptedException {
    Vector merged = VectorWritable.merge(vectors.iterator()).get();
    context.write(key, new VectorWritable(new SequentialAccessSparseVector(
        merged)));
  }
}