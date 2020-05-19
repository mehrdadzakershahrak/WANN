import gym
class CartPoleSwingUpEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.g = 9.82  #gravity
        self.m_c = 0.5 #cart mass
        self.m_p = 0.5  #pendulum mass
        self.total_m = (self.m_p + self.m_c)
        self.l = 0.6 #pole's length
        self.m_p_l = (self.m_p*self.l)
        self.force_mag = 10.0
        self.dt = 0.01  #seconds between state updates
        self.b = 0.1  #friction coefficient
        self.t = 0 #timestep
        self.t_limit = 3000
        self.hard_mode = False        
        self.theta_threshold_radians = 12 * 2 * math.pi / 360  #Angle at which to fail the episode
        self.x_threshold = 2.4
        self.action_space = 1
        self.observation_space = 5
        self.state = None

    def step(self, action):
        if (action < -1.0): action = -1.0
        if (action > 1.0): action = 1.0
        action *= self.force_mag
        state = self.state
        x = state[0]
        x_dot = state[1]
        theta = state[2]
        theta_dot = state[3]
        s = math.sin(theta)
        c = math.cos(theta)
        xdot_update = (-2*self.m_p_l*(theta_dot*theta_dot)*s + 3*self.m_p*self.g*s*c + 4*action - 4*self.b*x_dot)/(4*self.total_m - 3*self.m_p*c*c)
        thetadot_update = (-3*self.m_p_l*(theta_dot*theta_dot)*s*c + 6*self.total_m*self.g*s + 6*(action - self.b*x_dot)*c)/(4*self.l*self.total_m - 3*self.m_p_l*c*c)
        x = x + x_dot*self.dt
        theta = theta + theta_dot*self.dt
        x_dot = x_dot + xdot_update*self.dt
        theta_dot = theta_dot + thetadot_update*self.dt
        self.state = [x,x_dot,theta,theta_dot]
        done = False
        if ((x < -self.x_threshold) or (x > self.x_threshold)): done = True
        self.t += 1
        if (self.t >= self.t_limit): done = True
        reward_theta = (math.cos(theta)+1.0)/2.0
        reward_x = math.cos((x/self.x_threshold)*(math.pi/2.0))
        reward = reward_theta*reward_x
        obs = [x,x_dot,math.cos(theta),math.sin(theta),theta_dot]
        return [obs, reward, done]

    def reset(self):
        stdev = 0.1
        x = self.randn(0.0, stdev)
        x_dot = self.randn(0.0, stdev)
        theta = self.randn(math.pi, stdev)
        theta_dot = self.randn(0.0, stdev)
        x = self.randf(-1.0, 1.0)*self.x_threshold*0.75
        if (self.hard_mode):
            x = self.randf(-1.0, 1.0)*self.x_threshold*1.0
            x_dot = self.randf(-1.0, 1.0)*10.0*1.0
            theta = self.randf(-1.0, 1.0)*math.pi/2.0+math.pi
            theta_dot = self.randf(-1.0, 1.0)*10.0*1.0
        self.state = [x, x_dot, theta, theta_dot]
        self.t = 0
        obs = [x,x_dot,math.cos(theta),math.sin(theta),theta_dot]
        return obs

    def render(self, mode='human'):
        return

        screen_width = 800
        world_width = 5  #max visible position of cart
        scale = screen_width/world_width
        carty = screen_width/8 #TOP OF CART (assume screen_width == screen_height * 4)
        polewidth = 6.0*screen_width/600
        polelen = scale*self.l  #0.6 or self.l
        cartwidth = 40.0*screen_width/600
        cartheight = 20.0*screen_width/600
        state = self.state
        x = state[0]
        x_dot = state[1]
        theta = state[2]
        theta_dot = state[3]
        cartx = x*scale+screen_width/2.0 #MIDDLE OF CART
        self.p.stroke(0)
        self.p.strokeWeight(0.5)
        #track
        self.p.line(screen_width/2 - self.x_threshold*scale, carty + cartheight/2 + cartheight/4 + 1, screen_width/2 + self.x_threshold*scale, carty + cartheight/2 + cartheight/4 + 1)
        l=-cartwidth/2
        r=cartwidth/2
        t=cartheight/2
        b=-cartheight/2
        #cart
        self.p.fill(255, 64, 64)
        self.p.push()
        self.p.translate(cartx, carty)
        self.polygon(self.p, [[l,b], [l,t], [r,t], [r,b]])
        self.p.pop()
        #L and R wheels
        self.p.fill(192)
        self.p.circle(cartx-cartwidth/2, carty+cartheight/2, cartheight/2)
        self.p.circle(cartx+cartwidth/2, carty+cartheight/2, cartheight/2)
        #pole
        l=-polewidth/2
        r=polewidth/2
        t=polelen-polewidth/2
        b=-polewidth/2
        self.p.fill(64, 64, 255)
        self.p.push()
        self.p.translate(cartx, carty)
        self.p.rotate(math.pi-theta)
        self.polygon(self.p, [[l,b], [l,t], [r,t], [r,b]])
        self.p.pop()
        #axle
        self.p.fill(48)
        self.p.circle(cartx, carty, polewidth) #note: diameter, not radius.

    def polygon(self, p, points):
        p.beginShape()
        N = points.length
        for i in range(N):
            x = points[i][0]
            y = points[i][1]
            p.vertex(x, y)
        p.endShape(p.CLOSE)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def gaussRandom(self):
        if not hasattr(self, 'return_v'):
            self.return_v = False
            self.v_val = 0.0

        if(self.return_v):
            self.return_v = False
            return self.v_val

        u = 2*random.random()-1
        v = 2*random.random()-1
        r = u*u + v*v
        if(r == 0 or r > 1):
            return self.gaussRandom()
        c = math.sqrt(-2*math.log(r)/r)
        v_val = v*c  #cache this
        self.return_v = True
        return u*c

    def randf(self, a, b):
        return random.random()*(b-a)+a

    def randi(self, a, b):
        return math.floor(random.random()*(b-a)+a)

    def randn(self, mu, std):
        return mu+self.gaussRandom()*std

    def birandn(self, mu1, mu2, std1, std2, rho):
        z1 = randn(0, 1)
        z2 = randn(0, 1)
        x = math.sqrt(1-rho*rho)*std1*z1 + rho*std1*z2 + mu1
        y = std2*z2 + mu2
        return [x, y]