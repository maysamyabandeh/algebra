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