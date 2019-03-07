package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete;

import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.primitives.Pair;

import com.mes.schedule.deepQL.env.ScheduleEnv;
import com.mes.schedule.domain.SOperationTask;
import com.mes.schedule.domain.SPartTask;
import com.mes.schedule.domain.ScheduleScheme;
import com.mes.schedule.rl.deepQL.DeepQLParameters;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.DQNPolicySchedule;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.policy.EpsGreedySchedule;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.Constants;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.ArrayUtil;
import java.util.ArrayList;
import java.util.List;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/18/16.
 *
 *         DQN or Deep Q-Learning in the Discrete domain
 *
 *         http://arxiv.org/abs/1312.5602
 *
 */
public abstract class QLearningDiscreteSchedule<O extends Encodable> extends QLearning<O, Integer, DiscreteSpace> {

	@Getter
	final private QLConfiguration configuration;
	@Getter
	final private DataManager dataManager;
	@Getter
	final private MDP<O, Integer, DiscreteSpace> mdp;
	@Getter
	final private IDQN currentDQN;
	@Getter
	@Setter
	private DQNPolicySchedule<O> policy;
	@Getter
	@Setter
	private EpsGreedy<O, Integer, DiscreteSpace> egPolicy;
	/**
	 * 自定义的调度贪婪策略
	 */
	@Getter
	@Setter
	private EpsGreedySchedule<O, Integer, DiscreteSpace> egSchedulePolicy;
	@Getter
	@Setter
	private IDQN targetDQN;
	private int lastAction;
	private INDArray history[] = null;
	private double accuReward = 0;
	private int lastMonitor = -Constants.MONITOR_FREQ;

	public QLearningDiscreteSchedule(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, QLConfiguration conf,
			DataManager dataManager, int epsilonNbStep) {
		super(conf);
		this.configuration = conf;
		this.mdp = mdp;
		this.dataManager = dataManager;
		currentDQN = dqn;
		targetDQN = dqn.clone();
		policy = new DQNPolicySchedule(getCurrentDQN());
		// 使用调度贪婪策略
		egSchedulePolicy = new EpsGreedySchedule(policy, mdp, conf.getUpdateStart(), epsilonNbStep, getRandom(),
				conf.getMinEpsilon(), this);
		mdp.getActionSpace().setSeed(conf.getSeed());
	}

	public void postEpoch() {

		if (getHistoryProcessor() != null)
			getHistoryProcessor().stopMonitor();

	}

	public void preEpoch() {
		history = null;
		lastAction = 0;
		accuReward = 0;

		if (getStepCounter() - lastMonitor >= Constants.MONITOR_FREQ && getHistoryProcessor() != null
				&& getDataManager().isSaveData()) {
			lastMonitor = getStepCounter();
			int[] shape = getMdp().getObservationSpace().getShape();
			getHistoryProcessor().startMonitor(
					getDataManager().getVideoDir() + "/video-" + getEpochCounter() + "-" + getStepCounter() + ".mp4",
					shape);
		}
	}

	/**
	 * Single step of training
	 * 
	 * @param obs
	 *            last obs
	 * @return relevant info for next step
	 */
	protected QLStepReturn<O> trainStep(O obs) {

		Integer action = null;
		INDArray input = getInput(obs);
		boolean isHistoryProcessor = getHistoryProcessor() != null;

		if (isHistoryProcessor)
			getHistoryProcessor().record(input);

		int skipFrame = isHistoryProcessor ? getHistoryProcessor().getConf().getSkipFrame() : 1;
		int historyLength = isHistoryProcessor ? getHistoryProcessor().getConf().getHistoryLength() : 1;
		int updateStart = getConfiguration().getUpdateStart()
				+ ((getConfiguration().getBatchSize() + historyLength) * skipFrame);

		Double maxQ = Double.NaN; // ignore if Nan for stats

		// if step of training, just repeat lastAction
		if (getStepCounter() % skipFrame != 0) {
			action = lastAction;
		} else {
			if (history == null) {
				if (isHistoryProcessor) {
					getHistoryProcessor().add(input);
					history = getHistoryProcessor().getHistory();
				} else
					history = new INDArray[] { input };
			}
			// concat the history into a single INDArray input
			INDArray hstack = Transition.concat(Transition.dup(history));
			if (isHistoryProcessor) {
				hstack.muli(1.0 / getHistoryProcessor().getScale());
			}

			// if input is not 2d, you have to append that the batch is 1 length
			// high
			if (hstack.shape().length > 2)
				hstack = hstack.reshape(Learning.makeShape(1, ArrayUtil.toInts(hstack.shape())));

			INDArray qs = getCurrentDQN().output(hstack);
			int maxAction = Learning.getMaxAction(qs);

			maxQ = qs.getDouble(maxAction);
			if (DeepQLParameters.actionSelection == 0) {
				action = getEgSchedulePolicy().nextAction(hstack);
			} else {
				// 选择工序
				if (getMdp() instanceof ScheduleEnv) {
					ScheduleEnv scheduleEnv = (ScheduleEnv) getMdp();
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
					action = getEgSchedulePolicy().nextAction(hstack, actionsAtState);
				}
			}
		}

		lastAction = action;

		StepReply<O> stepReply = getMdp().step(action);

		accuReward += stepReply.getReward() * configuration.getRewardFactor();

		// if it's not a skipped frame, you can do a step of training
		if (getStepCounter() % skipFrame == 0 || stepReply.isDone()) {

			INDArray ninput = getInput(stepReply.getObservation());
			if (isHistoryProcessor)
				getHistoryProcessor().add(ninput);

			INDArray[] nhistory = isHistoryProcessor ? getHistoryProcessor().getHistory() : new INDArray[] { ninput };

			Transition<Integer> trans = new Transition(history, action, accuReward, stepReply.isDone(), nhistory[0]);
			// 如果就绪任务集合中只存在一个任务，则不需要将此类变迁存至经验回放中
			if (getMdp() instanceof ScheduleEnv) {
				/*
				 * ScheduleEnv scheduleEnv = (ScheduleEnv) getMdp(); if
				 * (scheduleEnv.getScheme().getReadyTaskS() != null &&
				 * scheduleEnv.getScheme().getReadyTaskS().size() > 1 ||
				 * DeepQLParameters.delayReward) {
				 */
				getExpReplay().store(trans);
				// }
			} else {
				getExpReplay().store(trans);
			}

			if (getStepCounter() > updateStart) {
				Pair<INDArray, INDArray> targets = setTarget(getExpReplay().getBatch());
				getCurrentDQN().fit(targets.getFirst(), targets.getSecond());
			}

			history = nhistory;
			accuReward = 0;
		}

		return new QLStepReturn<O>(maxQ, getCurrentDQN().getLatestScore(), stepReply);

	}

	protected Pair<INDArray, INDArray> setTarget(ArrayList<Transition<Integer>> transitions) {
		if (transitions.size() == 0)
			throw new IllegalArgumentException("too few transitions");

		int size = transitions.size();

		int[] shape = getHistoryProcessor() == null ? getMdp().getObservationSpace().getShape()
				: getHistoryProcessor().getConf().getShape();
		int[] nshape = makeShape(size, shape);
		INDArray obs = Nd4j.create(nshape);
		INDArray nextObs = Nd4j.create(nshape);
		int[] actions = new int[size];
		boolean[] areTerminal = new boolean[size];

		for (int i = 0; i < size; i++) {
			Transition<Integer> trans = transitions.get(i);
			areTerminal[i] = trans.isTerminal();
			actions[i] = trans.getAction();

			INDArray[] obsArray = trans.getObservation();
			if (obs.rank() == 2) {
				obs.putRow(i, obsArray[0]);
			} else {
				for (int j = 0; j < obsArray.length; j++) {
					obs.put(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.point(j) }, obsArray[j]);
				}
			}

			INDArray[] nextObsArray = Transition.append(trans.getObservation(), trans.getNextObservation());
			if (nextObs.rank() == 2) {
				nextObs.putRow(i, nextObsArray[0]);
			} else {
				for (int j = 0; j < nextObsArray.length; j++) {
					nextObs.put(new INDArrayIndex[] { NDArrayIndex.point(i), NDArrayIndex.point(j) }, nextObsArray[j]);
				}
			}
		}
		if (getHistoryProcessor() != null) {
			obs.muli(1.0 / getHistoryProcessor().getScale());
			nextObs.muli(1.0 / getHistoryProcessor().getScale());
		}

		INDArray dqnOutputAr = dqnOutput(obs);

		INDArray dqnOutputNext = dqnOutput(nextObs);
		INDArray targetDqnOutputNext = null;

		INDArray tempQ = null;
		INDArray getMaxAction = null;
		if (getConfiguration().isDoubleDQN()) {
			targetDqnOutputNext = targetDqnOutput(nextObs);
			getMaxAction = Nd4j.argMax(dqnOutputNext, 1);
		} else {
			tempQ = Nd4j.max(dqnOutputNext, 1);
		}

		for (int i = 0; i < size; i++) {
			double yTar = transitions.get(i).getReward();
			if (!areTerminal[i]) {
				double q = 0;
				if (getConfiguration().isDoubleDQN()) {
					q += targetDqnOutputNext.getDouble(i, getMaxAction.getInt(i));
				} else
					q += tempQ.getDouble(i);

				yTar += getConfiguration().getGamma() * q;

			}

			double previousV = dqnOutputAr.getDouble(i, actions[i]);
			double lowB = previousV - getConfiguration().getErrorClamp();
			double highB = previousV + getConfiguration().getErrorClamp();
			double clamped = Math.min(highB, Math.max(yTar, lowB));

			dqnOutputAr.putScalar(i, actions[i], clamped);
		}

		return new Pair(obs, dqnOutputAr);
	}

}
