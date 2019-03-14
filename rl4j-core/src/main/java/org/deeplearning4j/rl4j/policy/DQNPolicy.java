/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.rl4j.policy;

import lombok.AllArgsConstructor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/18/16.
 *
 *         DQN policy returns the action with the maximum Q-value as evaluated
 *         by the dqn model
 */
@AllArgsConstructor
public class DQNPolicy<O extends Encodable> extends Policy<O, Integer> {

	final private IDQN dqn;

	public static <O extends Encodable> DQNPolicy<O> load(String path) throws IOException {
		return new DQNPolicy<O>(DQN.load(path));
	}

	public IDQN getNeuralNet() {
		return dqn;
	}

	public Integer nextAction(INDArray input) {
		INDArray output = dqn.output(input);
		return Learning.getMaxAction(output);
	}

	/**
	 * 带有可选行为集合的贪婪行为
	 */
	public Integer nextAction(INDArray input, List<Integer> actionsAtState) {
		INDArray output = dqn.output(input);
		List<Integer> indList = new ArrayList<Integer>();
		int ind = -1;
		double maxValue = Double.NEGATIVE_INFINITY;
		for (Integer actionIndex : actionsAtState) {
			if (output.getDouble(actionIndex) > maxValue) {
				ind = actionIndex;
				maxValue = output.getDouble(actionIndex);
				// 有更大值则先清空列表再放入元素
				indList.clear();
				indList.add(ind);
			}
			// 相等则直接放入
			if (output.getDouble(actionIndex) == maxValue) {
				indList.add(actionIndex);
			}
		}
		if (indList.size() > 0) {
			return indList.get(new Random().nextInt(indList.size()));
		}
		return ind;
	}

	public void save(String filename) throws IOException {
		dqn.save(filename);
	}

}
