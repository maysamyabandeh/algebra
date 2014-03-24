/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
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

import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.classification.InterfaceStability;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

/** A WritableComparable for matrix elements. */
@InterfaceAudience.Public
@InterfaceStability.Stable
public class ElementWritable implements WritableComparable<ElementWritable> {
  private int row;
  private int col;

  public ElementWritable() {}

  /** Set the value of this IntWritable. */
  public void setRow(int row) { this.row = row; }
  public void setCol(int col) { this.col = col; }

  /** Return the value of this IntWritable. */
  public int getRow() { return row; }
  public int getCol() { return col; }

  @Override
  public void readFields(DataInput in) throws IOException {
    row = in.readInt();
    col = in.readInt();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(row);
    out.writeInt(col);
  }

  /** Returns true iff <code>o</code> is a ElementWritable with the same value. */
  @Override
  public boolean equals(Object o) {
    if (!(o instanceof ElementWritable))
      return false;
    ElementWritable other = (ElementWritable)o;
    return this.row == other.row;
  }

  @Override
  public int hashCode() {
    return row;
  }

  /** Compares two IntWritables. */
  @Override
  public int compareTo(ElementWritable that) {
    return (this.row<that.row ? -1 : (this.row==that.row ? 0 : 1));
  }

  @Override
  public String toString() {
    return "<"+row+","+col+">";
  }

  /** A Comparator optimized for IntWritable. */ 
  public static class Comparator extends WritableComparator {
    public Comparator() {
      super(ElementWritable.class);
    }
    
    @Override
    public int compare(byte[] b1, int s1, int l1,
                       byte[] b2, int s2, int l2) {
      int thisValue = readInt(b1, s1);
      int thatValue = readInt(b2, s2);
      return (thisValue<thatValue ? -1 : (thisValue==thatValue ? 0 : 1));
    }
  }

  static {                                        // register this comparator
    WritableComparator.define(ElementWritable.class, new Comparator());
  }
}

