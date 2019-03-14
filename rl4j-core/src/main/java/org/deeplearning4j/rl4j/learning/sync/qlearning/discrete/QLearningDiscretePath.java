package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete;

import lombok.Getter;
import lombok.Setter;
import model.Path;
import model.PathEnv;
import model.Point;

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
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.policy.EpsGreedyPath;
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

import javax.swing.text.html.HTMLDocument.HTMLReader.ParagraphAction;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/18/16.
 *
 *         DQN or Deep Q-Learning in the Discrete domain
 *
 *         http://arxiv.org/abs/1312.5602
 *
 */
public abstract class QLearningDiscretePath<O extends Encodable> extends QLearning<O, Integer, DiscreteSpace> {

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
	private DQNPolicy<O> policy;
	@Getter
	@Setter
	private EpsGreedy<O, Integer, DiscreteSpace> egPolicy;
	/**
	 * 自定义的路径贪婪策略
	 */
	@Getter
	@Setter
	private EpsGreedyPath<O, Integer, DiscreteSpace> egPathPolicy;
	@Getter
	@Setter
	private IDQN targetDQN;
	private int lastAction;
	private INDArray history[] = null;
	private double accuReward = 0;
	private int lastMonitor = -Constants.MONITOR_FREQ;

	public QLearningDiscretePath(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, QLConfiguration conf,
			DataManager dataManager, int epsilonNbStep) {
		super(conf);
		this.configuration = conf;
		this.mdp = mdp;
		this.dataManager = dataManager;
		currentDQN = dqn;
		targetDQN = dqn.clone();
		policy = new DQNPolicy(getCurrentDQN());
		egPathPolicy = new EpsGreedyPath(policy, mdp, conf.getUpdateStart(), epsilonNbStep, getRandom(),
				conf.getMinEpsilon(), this);
		egPolicy = new EpsGreedy(policy, mdp, conf.getUpdateStart(), epsilonNbStep, getRandom(), conf.getMinEpsilon(),
				this);
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

			PathEnv pathEnv = (PathEnv) getMdp();
			List<Integer> actionsAtState = new ArrayList<Integer>();
			Point currentPoint = pathEnv.getCurrentPoint();
			for (Path path : currentPoint.getSuccPathSet()) {
				actionsAtState.add((Integer) pathEnv.getPathsMap().get(path));
			}
			action = getEgPathPolicy().nextAction(hstack, actionsAtState);

			INDArray qs = getCurrentDQN().output(hstack);
			// 这个地方要考虑可选行为集合，将非可选行为位置上的值置为最小值
			INDArray qs_ = qs.dup();
			for (int i = 0; i < actionsAtState.size(); i++) {
				for (int col = 0; col < qs_.columns(); col++) {
					if (!actionsAtState.contains(col)) {
						qs_.putScalar(0, col, -1 * Double.MAX_VALUE);
					}
				}
			}
			int maxAction = Learning.getMaxAction(qs_);

			maxQ = qs_.getDouble(maxAction);
			// if (DeepQLParameters.actionSelection == 0) {
			// action = getEgPolicy().nextAction(hstack);
			// } else {
			// }

			// 选择路径
			// if (getMdp() instanceof PathEnv) {
			// }
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
		// 实际上是有可选行为的，所以创建一个新的INDArray dqnOutputNext_
		INDArray dqnOutputNext_ = dqnOutputNext.dup();
		PathEnv pathEnv = (PathEnv) this.mdp;
		for (int i = 0; i < size; i++) {
			// 获取对应状态
			// double nextState = nextObs.getDouble(i, 0);
			// 根据动作确定路径
			Path pathAction = pathEnv.getPath(actions[i]);
			// 使用路径选择下一节点
			Point nextPoint = pathAction.getSuccPoint();// (Point)
			// pathEnv.getPointsMap().get((int)
			// nextState);
			// 用于存储可选路径的编号，这些位置下的数值不变，其他位置需要设置成最小值
			List<Integer> pathsIndex = new ArrayList<Integer>();
			for (Path path : nextPoint.getSuccPathSet()) {
				pathsIndex.add((Integer) pathEnv.getPathsMap().get(path));
			}
			for (int col = 0; col < dqnOutputNext_.columns(); col++) {
				if (!pathsIndex.contains(col)) {
					dqnOutputNext_.putScalar(i, col, -1 * Double.MAX_VALUE);
				}
			}
		}
		if (getConfiguration().isDoubleDQN()) {
			targetDqnOutputNext = targetDqnOutput(nextObs);
			getMaxAction = Nd4j.argMax(dqnOutputNext_, 1);
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
