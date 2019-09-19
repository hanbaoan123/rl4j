package org.deeplearning4j.rl4j.policy;

import lombok.AllArgsConstructor;
import model.Path;
import model.PathEnv;
import model.Point;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.mes.schedule.deepQL.env.InstanceEnv;
import com.mes.schedule.deepQL.env.ScheduleEnv;
import com.mes.schedule.domain.SOperationTask;
import com.mes.schedule.domain.SPartTask;
import com.mes.schedule.domain.ScheduleScheme;
import com.mes.schedule.rl.deepQL.DeepQLParameters;
import org.nd4j.linalg.util.ArrayUtil;
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
public class DQNPolicyInstance<O extends Encodable> extends Policy<O, Integer> {

	final private IDQN dqn;

	public static <O extends Encodable> DQNPolicyInstance<O> load(String path) throws IOException {
		return new DQNPolicyInstance<O>(DQN.load(path));
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

	@Override
	public <AS extends ActionSpace<Integer>> double play(MDP<O, Integer, AS> mdp, IHistoryProcessor hp) {
		getNeuralNet().reset();
		Learning.InitMdp<O> initMdp = Learning.initMdp(mdp, hp);
		if (mdp instanceof InstanceEnv) {
			InstanceEnv instanceEnv = (InstanceEnv) mdp;
			instanceEnv.getInstance().setWirteDynamic(true);
		}
		O obs = initMdp.getLastObs();

		double reward = initMdp.getReward();

		Integer lastAction = mdp.getActionSpace().noOp();
		Integer action = null;
		int step = initMdp.getSteps();
		INDArray[] history = null;

		while (!mdp.isDone()) {

			INDArray input = Learning.getInput(mdp, obs);
			boolean isHistoryProcessor = hp != null;

			if (isHistoryProcessor)
				hp.record(input);

			int skipFrame = isHistoryProcessor ? hp.getConf().getSkipFrame() : 1;

			if (step % skipFrame != 0) {
				action = lastAction;
			} else {

				if (history == null) {
					if (isHistoryProcessor) {
						hp.add(input);
						history = hp.getHistory();
					} else
						history = new INDArray[] { input };
				}
				INDArray hstack = Transition.concat(history);
				if (isHistoryProcessor) {
					hstack.muli(1.0 / hp.getScale());
				}
				if (getNeuralNet().isRecurrent()) {
					// flatten everything for the RNN
					hstack = hstack.reshape(Learning.makeShape(1, ArrayUtil.toInts(hstack.shape()), 1));
				} else {
					if (hstack.shape().length > 2)
						hstack = hstack.reshape(Learning.makeShape(1, ArrayUtil.toInts(hstack.shape())));
				}
				if (DeepQLParameters.actionSelection == 0) {
					action = nextAction(hstack);
				} else {
					// 选择工序
					if (mdp instanceof ScheduleEnv) {
						ScheduleEnv scheduleEnv = (ScheduleEnv) mdp;
						ScheduleScheme scheme = scheduleEnv.getScheme();
						List<Integer> actionsAtState = new ArrayList<Integer>();
						int partIndex = 0;
						for (SPartTask partTask : scheme.getSchedulePartTasks()) {
							for (int i = partTask.getOperationTaskList().size() - 1; i >= 0; i--) {
								SOperationTask operationTask = partTask.getOperationTaskList().get(i);
								if (operationTask.getAssignState() == SOperationTask.ASSNSTATE_WAITING) {
									actionsAtState.add(partIndex);
									break;
								}
							}
							partIndex++;
						}
						action = nextAction(hstack, actionsAtState);
					}
				}
			}
			lastAction = action;

			StepReply<O> stepReply = mdp.step(action);
			reward += stepReply.getReward();

			if (isHistoryProcessor)
				hp.add(Learning.getInput(mdp, stepReply.getObservation()));

			history = isHistoryProcessor ? hp.getHistory()
					: new INDArray[] { Learning.getInput(mdp, stepReply.getObservation()) };
			step++;
		}
		return reward;
	}
}
