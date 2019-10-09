package org.deeplearning4j.rl4j.learning.sync.qlearning.discrete;

import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.DQNFactory;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.util.DataManager;
import org.deeplearning4j.rl4j.util.DataManagerTrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/6/16.
 */
public class QLearningDiscreteDensePath<O extends Encodable> extends QLearningDiscretePath<O> {


    @Deprecated
    public QLearningDiscreteDensePath(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, QLearning.QLConfiguration conf,
                    IDataManager dataManager) {
        this(mdp, dqn, conf);
        addListener(new DataManagerTrainingListener(dataManager));
    }
    public QLearningDiscreteDensePath(MDP<O, Integer, DiscreteSpace> mdp, IDQN dqn, QLearning.QLConfiguration conf) {
        super(mdp, dqn, conf, conf.getEpsilonNbStep());
    }

    @Deprecated
    public QLearningDiscreteDensePath(MDP<O, Integer, DiscreteSpace> mdp, DQNFactory factory,
                    QLearning.QLConfiguration conf, IDataManager dataManager) {
        this(mdp, factory.buildDQN(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf,
                        dataManager);
    }
    public QLearningDiscreteDensePath(MDP<O, Integer, DiscreteSpace> mdp, DQNFactory factory,
                                  QLearning.QLConfiguration conf) {
        this(mdp, factory.buildDQN(mdp.getObservationSpace().getShape(), mdp.getActionSpace().getSize()), conf);
    }

    @Deprecated
    public QLearningDiscreteDensePath(MDP<O, Integer, DiscreteSpace> mdp, DQNFactoryStdDense.Configuration netConf,
                    QLearning.QLConfiguration conf, IDataManager dataManager) {
        this(mdp, new DQNFactoryStdDense(netConf), conf, dataManager);
    }
    public QLearningDiscreteDensePath(MDP<O, Integer, DiscreteSpace> mdp, DQNFactoryStdDense.Configuration netConf,
                                  QLearning.QLConfiguration conf) {
        this(mdp, new DQNFactoryStdDense(netConf), conf);
    }

}
