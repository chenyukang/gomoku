#![allow(dead_code)]
use super::board::*;
use super::utils::*;
use rand::Rng;

type Id = usize;

pub struct Tree {
    descendants: Vec<Node>,
}

#[derive(Debug)]
pub struct Node {
    parent: Id,
    index: Id,
    children: Vec<Id>,
    visited_count: u32,
    win_count: u32,
    loss_count: u32,
    state: Board,
    untried_moves: Vec<Move>,
    player: u8,
    action: Option<Move>,
}

impl Tree {
    pub fn new() -> Self {
        Tree {
            descendants: vec![],
        }
    }

    pub fn new_node(
        &mut self,
        parent: Id,
        state: Board,
        player: u8,
        mv: Option<Move>,
    ) -> Option<&Node> {
        let id = self.descendants.len();
        self.descendants
            .push(Node::new(id, parent, state, player, mv));
        //println!("parent: {} id: {}", parent, id);
        if parent != id {
            self.descendants[parent].children.push(id);
        }
        self.descendants.get(id)
    }

    pub fn get_node(&self, id: Id) -> Option<&Node> {
        self.descendants.get(id)
    }

    pub fn expand(&mut self, index: Id) -> Option<&Node> {
        //println!("{} expand:", index);
        let parent = self.descendants.get(index).unwrap();
        let player = parent.player;
        let mv = parent.untried_moves.first().unwrap().clone();
        self.descendants[index].untried_moves.remove(0);
        let mut board = self.descendants[index].state.clone();
        board.place(mv.x, mv.y, player);
        self.new_node(index, board, cfg::opponent(player), Some(mv))
    }

    pub fn backpropagete(&mut self, index: Id, winner: Option<u8>) {
        self.descendants[index].visited_count += 1;
        let player = self.descendants[index].player;
        if winner == Some(player) {
            self.descendants[index].loss_count += 1;
        } else if winner == Some(cfg::opponent(player)) {
            self.descendants[index].win_count += 1;
        }
        if !self.descendants[index].is_root() {
            self.backpropagete(self.descendants[index].parent, winner);
        }
    }

    fn rollout_policy(&self, moves: &Vec<Move>) -> Move {
        moves[0].clone()
        /*  let mut rng = rand::thread_rng();
        let l = std::cmp::min(moves.len(), 4);
        let idx = rng.gen_range(0..l);
        moves[idx].clone() */
    }

    pub fn rollout(&mut self, index: Id) -> Option<u8> {
        let mut current_state = self.descendants[index].state.clone();
        let mut player = self.descendants[index].player;
        //println!("now player: {}", player);
        loop {
            let w = current_state.any_winner();
            if w.is_some() {
                return w;
            } else {
                let moves = current_state.gen_ordered_moves_all(player);
                let mv = self.rollout_policy(&moves);
                //println!("mv: {:?}", mv);
                current_state.place(mv.x, mv.y, player);
                player = cfg::opponent(player);
                //current_state.print();
                //println!();
            }
        }
    }

    pub fn best_child(&self, index: usize) -> usize {
        let c_param = 1.4;
        let mut res = 0;
        let mut cur_max = f64::MIN;
        let node = &self.descendants[index];
        for i in 0..node.children.len() {
            let c = &self.descendants[node.children[i]];
            let r = (2.0 * node.n().ln() / c.n()).sqrt();
            let v = (c.q() / c.n()) + c_param * r;
            //println!("[{}]: q({}) n({}) => v({})", i, c.q(), c.n(), v);
            if v > cur_max {
                cur_max = v;
                res = c.index;
            }
        }
        return res;
    }

    pub fn best_move(&self, index: usize) -> usize {
        let mut res = 0;
        let mut cur_max = f64::MIN;
        let node = &self.descendants[index];
        for i in 0..node.children.len() {
            let c = &self.descendants[node.children[i]];
            let v = c.win_count as f64 / c.visited_count as f64;
            if v > cur_max {
                cur_max = v;
                res = c.index;
            }
        }
        return res;
    }
}

impl Node {
    pub fn new(index: Id, parent: Id, state: Board, player: u8, mv: Option<Move>) -> Self {
        let mut s = Node {
            visited_count: 0,
            win_count: 0,
            loss_count: 0,
            state: state.clone(),
            untried_moves: vec![],
            children: vec![],
            player: player,
            parent: parent,
            index: index,
            action: mv,
        };
        s.untried_moves = s.state.gen_ordered_moves_all(player);
        s
    }

    pub fn is_root(&self) -> bool {
        self.parent == self.index
    }

    pub fn n(&self) -> f64 {
        self.visited_count as f64
    }

    pub fn q(&self) -> f64 {
        (self.win_count as i32 - self.loss_count as i32) as f64
    }

    pub fn is_fully_expanded(&self) -> bool {
        self.untried_moves.len() == 0
    }

    pub fn is_terminal_node(&self) -> bool {
        self.state.any_winner() != None
    }
}

pub struct MonteCarlo {
    tree: Tree,
    simulate_count: u32,
}

impl MonteCarlo {
    pub fn new(state: Board, player: u8, simulate_count: u32) -> Self {
        let mut s = Self {
            tree: Tree::new(),
            simulate_count: simulate_count,
        };
        s.tree.new_node(0, state, player, None);
        s
    }

    fn tree_policy(&mut self) -> usize {
        let mut cur = 0;
        loop {
            if self.tree.descendants[cur].is_terminal_node() {
                //println!("is terminate");
                break;
            }
            if !self.tree.descendants[cur].is_fully_expanded() {
                //println!("{} expand try ...", cur);
                let n = self.tree.expand(cur);
                cur = n.unwrap().index;
                break;
            } else {
                cur = self.tree.best_child(cur);
            }
        }
        return cur;
    }

    pub fn search_move(&mut self) -> Move {
        for i in 0..self.simulate_count {
            let v = self.tree_policy();
            println!(
                "{} {} : rollout {}",
                (i as f32 * 100.0) / self.simulate_count as f32,
                v,
                self.tree.descendants.len()
            );
            let r = self.tree.rollout(v);
            self.tree.backpropagete(v, r);
        }
        let best = self.tree.best_move(0);
        println!(
            "len: {} best: {}",
            self.tree.descendants[0].children.len(),
            best
        );
        let mut score = vec![];
        let mut moves = vec![];
        for x in self.tree.descendants[0].children.iter() {
            //println!("x: {}", x);
            moves.push(self.tree.descendants[*x].action.unwrap());
            score.push(vec![
                self.tree.descendants[*x].win_count as usize,
                self.tree.descendants[*x].loss_count as usize,
            ]);
            //self.tree.descendants[*x].state.print();
            //println!("=============");
        }
        let res = self.tree.descendants[best].action.unwrap();
        println!("move: {:?}", res);
        self.tree.descendants[best]
            .state
            .print_debug(&moves, &score);
        println!("win: {}", self.tree.descendants[best].win_count);
        println!("loss: {}", self.tree.descendants[best].loss_count);
        println!("visited: {}", self.tree.descendants[best].visited_count);
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_create() {
        let mut board = Board::new_default();
        board.place(5, 5, 1);
        let mut root = Tree::new();
        root.new_node(0, board, 1, None).unwrap();
        assert_eq!(root.get_node(0).unwrap().parent, 0);

        assert_eq!(root.get_node(0).unwrap().untried_moves.len(), 8);

        root.expand(0);
        assert_eq!(root.get_node(0).unwrap().parent, 0);
        assert_eq!(root.get_node(0).unwrap().untried_moves.len(), 7);
        assert_eq!(root.get_node(1).unwrap().parent, 0);
        root.backpropagete(1, Some(2));

        assert_eq!(root.get_node(1).unwrap().untried_moves.len(), 12);
        root.expand(1);
        assert_eq!(root.get_node(1).unwrap().untried_moves.len(), 11);
    }

    #[test]
    fn test_tree() {
        let mut root = Tree::new();
        assert_eq!(root.descendants.len(), 0);
        let node = root.new_node(0, Board::new_default(), 1, None).unwrap();
        assert_eq!(node.parent, 0);
        assert_eq!(node.index, 0);
        assert_eq!(node.is_root(), true);
        let node_2 = root.new_node(0, Board::new_default(), 2, None).unwrap();
        assert_eq!(node_2.parent, 0);
        assert_eq!(root.get_node(0).unwrap().children.len(), 1);
        assert_eq!(root.get_node(1).unwrap().children.len(), 0);
        root.backpropagete(1, Some(1));
        assert_eq!(root.get_node(0).unwrap().loss_count, 1);
        assert_eq!(root.get_node(0).unwrap().win_count, 0);
    }

    #[test]
    fn test_rollout() {
        let mut board = Board::new_default();
        board.place(5, 5, 1);
        let mut root = Tree::new();
        root.new_node(0, board, 2, None).unwrap();
        assert_eq!(root.get_node(0).unwrap().parent, 0);
        assert_eq!(root.get_node(0).unwrap().player, 2);

        let res = root.rollout(0);
        assert_eq!(res.is_some(), true);
    }

    #[test]
    fn test_monte_carlo() {
        let mut board = Board::new_default();
        board.place(7, 7, 1);
        let mut monte_carlo = MonteCarlo::new(board, 2, 20);
        assert_eq!(monte_carlo.tree.descendants[0].is_fully_expanded(), false);
        let mv = monte_carlo.search_move();
        assert_eq!(mv.x == 8 || mv.y == 8, true);
        println!("{:?}", mv);
    }
}
