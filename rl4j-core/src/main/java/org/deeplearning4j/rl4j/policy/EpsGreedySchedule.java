package org.deeplearning4j.rl4j.policy;

import lombok.AllArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.rl4j.learning.StepCountable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;

import com.mes.schedule.deepQL.env.ScheduleDiscreteActionSpace;
import com.mes.schedule.deepQL.env.ScheduleEnv;

import java.util.List;
import org.nd4j.linalg.api.rng.Random;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 1/8/19.
 *
 *         An epsilon greedy policy choose the next action - randomly with
 *         epsilon probability - deleguate it to constructor argument 'policy'
 *         with (1-epsilon) probability.
 *
 *         epislon is annealed to minEpsilon over epsilonNbStep steps
 *         调度贪婪策略，在随机搜索时是基于一定知识的启发式搜索
 *
 */
@AllArgsConstructor
@Slf4j
@Setter
public class EpsGreedySchedule<O extends Encodable, A, AS extends ActionSpace<A>> extends Policy<O, A> {

	private Policy<O, A> policy;
	final private MDP<O, A, AS> mdp;
	final private int updateStart;
	final private int epsilonNbStep;
	final private Random rnd;
	final private float minEpsilon;
	final private StepCountable learning;

	public NeuralNet getNeuralNet() {
		return policy.getNeuralNet();
	}

	/**
	 * 此方法进行了一定改进，为了保证选择工序时能选择到有效的
	 */
	public A nextAction(INDArray input, MDP<O, A, AS> mdp) {

		float ep = getEpsilon();
		if (learning.getStepCounter() % 500 == 1)
			log.info("EP: " + ep + " " + learning.getStepCounter());
		if (rnd.nextFloat() > ep)
			return policy.nextAction(input);
		else
			return mdp.getActionSpace().randomAction();

	}

	public A nextAction(INDArray input) {
		float ep = getEpsilon();
		if (learning.getStepCounter() % 500 == 1)
			log.info("EP: " + ep + " " + learning.getStepCounter());
		if (rnd.nextFloat() > ep)
			return policy.nextAction(input);
		else
			return mdp.getActionSpace().randomAction();
	}

	public float getEpsilon() {
		return Math.min(1f, Math.max(minEpsilon, 1f - (learning.getStepCounter() - updateStart) * 1f / epsilonNbStep));
	}

	/**
	 * 加入了可选行为集合 hba
	 * 
	 * @param hstack
	 * @param actionsAtState
	 * @return 下午3:35:48
	 */
	public A nextAction(INDArray input, List<Integer> actionsAtState) {
		float ep = getEpsilon();
		if (learning.getStepCounter() % 500 == 1)
			log.info("EP: " + ep + " " + learning.getStepCounter());
		if (rnd.nextFloat() > ep)
			return policy.nextAction(input, actionsAtState);
		else
			return (A) ((ScheduleDiscreteActionSpace) mdp.getActionSpace()).randomAction(actionsAtState);
	}
}
