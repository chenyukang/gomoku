#![allow(dead_code)]
use super::board::*;
use super::utils::*;

type Id = usize;

pub struct Tree {
    descendants: Vec<Node>,
}

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
}

impl Tree {
    pub fn new() -> Self {
        Tree {
            descendants: vec![],
        }
    }

    pub fn new_node(&mut self, parent: Id, state: Board, player: u8) -> Option<&Node> {
        let id = self.descendants.len();
        self.descendants.push(Node::new(id, parent, state, player));
        if parent != id {
            self.descendants[parent].children.push(id);
        }
        self.descendants.get(id)
    }

    pub fn get_node(&self, id: Id) -> Option<&Node> {
        self.descendants.get(id)
    }

    pub fn expand(&mut self, index: Id) -> Option<&Node> {
        let parent = self.descendants.get(index).unwrap();
        let player = parent.player;
        let mv = parent.untried_moves.first().unwrap().clone();
        self.descendants[index].untried_moves.remove(0);
        let mut board = self.descendants[index].state.clone();
        board.place(mv.x, mv.y, player);
        self.new_node(index, board, player)
    }

    pub fn backpropagete(&mut self, index: Id, winner: Option<u8>) {
        self.descendants[index].visited_count += 1;
        let player = self.descendants[index].player;
        if winner == Some(player) {
            self.descendants[index].win_count += 1;
        } else if winner == Some(cfg::opponent(player)) {
            self.descendants[index].loss_count += 1;
        }
        if !self.descendants[index].is_root() {
            self.backpropagete(self.descendants[index].parent, winner);
        }
    }
}

impl Node {
    pub fn new(index: Id, parent: Id, state: Board, player: u8) -> Self {
        Self {
            visited_count: 0,
            win_count: 0,
            loss_count: 0,
            state: state.clone(),
            untried_moves: state.gen_possible_moves(),
            children: vec![],
            player: player,
            parent: parent,
            index: index,
        }
    }

    pub fn is_root(&self) -> bool {
        self.parent == self.index
    }

    pub fn n(&self) -> i32 {
        self.visited_count as i32
    }

    pub fn q(&self) -> i32 {
        self.win_count as i32 - self.loss_count as i32
    }

    pub fn is_terminal_node(&self) -> bool {
        self.state.any_winner() == None
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
        root.new_node(0, board, 1).unwrap();
        assert_eq!(root.get_node(0).unwrap().parent, 0);

        assert_eq!(root.get_node(0).unwrap().untried_moves.len(), 8);

        root.expand(0);
        assert_eq!(root.get_node(0).unwrap().parent, 0);
        assert_eq!(root.get_node(0).unwrap().untried_moves.len(), 7);
        assert_eq!(root.get_node(1).unwrap().parent, 0);
        root.backpropagete(1, Some(2));
    }

    #[test]
    fn test_tree() {
        let mut root = Tree::new();
        assert_eq!(root.descendants.len(), 0);
        let node = root.new_node(0, Board::new_default(), 1).unwrap();
        assert_eq!(node.parent, 0);
        assert_eq!(node.index, 0);
        assert_eq!(node.is_root(), true);
        let node_2 = root.new_node(0, Board::new_default(), 2).unwrap();
        assert_eq!(node_2.parent, 0);
        assert_eq!(root.get_node(0).unwrap().children.len(), 1);
        root.backpropagete(1, Some(1));
        assert_eq!(root.get_node(0).unwrap().loss_count, 1);
        assert_eq!(root.get_node(0).unwrap().win_count, 0);
    }
}
