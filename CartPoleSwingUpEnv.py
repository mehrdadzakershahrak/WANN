# Importing the gym module from OpenAI for reinforcement learning environments
import gym

# Defining a custom gym environment for a swing up cart pole problem
class CartPoleSwingUpEnv(gym.Env):
    # Metadata about the environment, including rendering modes and video frames per second
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    # The initialization function for the environment
    def __init__(self):
        # Physical parameters of the cart-pole system
        self.g = 9.82  #gravity
        self.m_c = 0.5 #cart mass
        self.m_p = 0.5  #pendulum mass
        self.total_m = (self.m_p + self.m_c)  # total mass
        self.l = 0.6 #pole's length
        self.m_p_l = (self.m_p*self.l)  # mass*length
        self.force_mag = 10.0  # force magnitude
        self.dt = 0.01  #seconds between state updates
        self.b = 0.1  #friction coefficient

        # Time-related parameters
        self.t = 0 #timestep
        self.t_limit = 3000  # maximum number of timesteps

        # Other parameters
        self.hard_mode = False  # a flag for difficulty level
        self.theta_threshold_radians = 12 * 2 * math.pi / 360  #Angle at which to fail the episode
        self.x_threshold = 2.4  # maximum x position
        self.action_space = 1  # size of action space
        self.observation_space = 5  # size of observation space
        self.state = None  # current state of the environment

    # Function to execute an action and update the environment state
    def step(self, action):
        # Ensure action is within valid range
        action = np.clip(action, -1, 1)
        # Scale action by force magnitude
        action *= self.force_mag

        # Get current state
        state = self.state
        x, x_dot, theta, theta_dot = state  # unpack state variables

        # Calculate new state variables using physics equations
        s = math.sin(theta)
        c = math.cos(theta)
        # Calculate updates for velocity and angular velocity
        xdot_update = (-2*self.m_p_l*(theta_dot*theta_dot)*s + 3*self.m_p*self.g*s*c + 4*action - 4*self.b*x_dot)/(4*self.total_m - 3*self.m_p*c*c)
        thetadot_update = (-3*self.m_p_l*(theta_dot*theta_dot)*s*c + 6*self.total_m*self.g*s + 6*(action - self.b*x_dot)*c)/(4*self.l*self.total_m - 3*self.m_p_l*c*c)
        # Apply updates to state variables
        x = x + x_dot*self.dt
        theta = theta + theta_dot*self.dt
        x_dot = x_dot + xdot_update*self.dt
        theta_dot = theta_dot + thetadot_update*self.dt
        # Update the state
        self.state = [x,x_dot,theta,theta_dot]

        # Check if episode is done
        done = False
        if ((x < -self.x_threshold) or (x > self.x_threshold)): done = True  # if cart position is out of bounds
        self.t += 1
                if (self.t >= self.t_limit): done = True  # if maximum timesteps reached

        # Calculate reward
        reward_theta = (math.cos(theta)+1.0)/2.0  # reward component based on pole angle
        reward_x = math.cos((x/self.x_threshold)*(math.pi/2.0))  # reward component based on cart position
        reward = reward_theta*reward_x  # total reward is product of components

        # Construct observation
        obs = [x, x_dot, math.cos(theta), math.sin(theta), theta_dot]

        return [obs, reward, done]  # return observation, reward and done flag

    # Function to reset the environment to initial state
    def reset(self):
        stdev = 0.1  # standard deviation for random initialization

        # Randomly initialize state variables
        x = self.randn(0.0, stdev)
        x_dot = self.randn(0.0, stdev)
        theta = self.randn(math.pi, stdev)
        theta_dot = self.randn(0.0, stdev)

        # Further randomization for hard mode
        if (self.hard_mode):
            x = self.randf(-1.0, 1.0)*self.x_threshold*1.0
            x_dot = self.randf(-1.0, 1.0)*10.0*1.0
            theta = self.randf(-1.0, 1.0)*math.pi/2.0+math.pi
            theta_dot = self.randf(-1.0, 1.0)*10.0*1.0

        self.state = [x, x_dot, theta, theta_dot]  # set state
        self.t = 0  # reset timestep

        # Return initial observation
        obs = [x, x_dot, math.cos(theta), math.sin(theta), theta_dot]
        return obs

    # Function to render the environment
    def render(self, mode='human'):
        # Rendering code is omitted for brevity, but would typically involve drawing the cart-pole system

    # Function to close the environment
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # Function to set the seed for random number generation
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

        def gaussRandom(self):
        """
        Function to generate Gaussian random numbers using the Box–Muller transform.
        It generates two independent standard normally distributed random numbers, but returns 
        only one at a time. The second number is cached for the next call.
        """
        if not hasattr(self, 'return_v'):  # if 'return_v' is not an attribute of 'self', initialize it
            self.return_v = False
            self.v_val = 0.0

        if(self.return_v):  # if the second random number is in cache, return it
            self.return_v = False
            return self.v_val

        # generate two uniform random numbers
        u = 2*random.random()-1
        v = 2*random.random()-1
        r = u*u + v*v
        # if r is 0 or greater than 1, get another pair of random numbers
        if(r == 0 or r > 1):
            return self.gaussRandom()
        c = math.sqrt(-2*math.log(r)/r)
        self.v_val = v*c  # cache the second random number
        self.return_v = True
        return u*c  # return the first random number

    def randf(self, a, b):
        """
        Function to generate a random floating point number between 'a' and 'b'.
        """
        return random.random()*(b-a)+a

    def randi(self, a, b):
        """
        Function to generate a random integer between 'a' and 'b'.
        """
        return math.floor(random.random()*(b-a)+a)

    def randn(self, mu, std):
        """
        Function to generate a random number with normal (Gaussian) distribution.
        'mu' is the mean and 'std' is the standard deviation of the distribution.
        """
        return mu+self.gaussRandom()*std

    def birandn(self, mu1, mu2, std1, std2, rho):
        """
        Function to generate a pair of random numbers with a bivariate normal distribution.
        'mu1' and 'mu2' are the means, 'std1' and 'std2' are the standard deviations, and
        'rho' is the correlation coefficient.
        """
        z1 = self.randn(0, 1)
        z2 = self.randn(0, 1)
        x = math.sqrt(1-rho*rho)*std1*z1 + rho*std1*z2 + mu1
        y = std2*z2 + mu2
        return [x, y]


