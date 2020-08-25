class UserService:
    def __init__(self, dao):
        self.__user_dao = dao

    def exists_user(self, username, password):
        return self.__user_dao.exists_user(username, password)

    def add_user(self, username, password):
        if self.exists_user(username, password):
            return False
        return self.__user_dao.add_user(username, password)

    def get_no_attempts(self, username):
        return self.__user_dao.get_attempts(username)

    def update_user_attempts(self, username):
        no_old_attempts = self.__user_dao.get_attempts(username)
        return self.__user_dao.update_attempts(username, no_old_attempts + 1)