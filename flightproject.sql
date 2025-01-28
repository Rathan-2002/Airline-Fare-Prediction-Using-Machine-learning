-- phpMyAdmin SQL Dump
-- version 5.1.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1:3306
-- Generation Time: Apr 03, 2023 at 12:10 PM
-- Server version: 5.7.36
-- PHP Version: 7.4.26

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `flightproject`
--
CREATE DATABASE IF NOT EXISTS `flightproject` DEFAULT CHARACTER SET latin1 COLLATE latin1_swedish_ci;
USE `flightproject`;

-- --------------------------------------------------------

--
-- Table structure for table `auth_group`
--

DROP TABLE IF EXISTS `auth_group`;
CREATE TABLE IF NOT EXISTS `auth_group` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(150) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `auth_group_permissions`
--

DROP TABLE IF EXISTS `auth_group_permissions`;
CREATE TABLE IF NOT EXISTS `auth_group_permissions` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `group_id` int(11) NOT NULL,
  `permission_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_group_permissions_group_id_permission_id_0cd325b0_uniq` (`group_id`,`permission_id`),
  KEY `auth_group_permissions_group_id_b120cbf9` (`group_id`),
  KEY `auth_group_permissions_permission_id_84c5c92e` (`permission_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `auth_permission`
--

DROP TABLE IF EXISTS `auth_permission`;
CREATE TABLE IF NOT EXISTS `auth_permission` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `content_type_id` int(11) NOT NULL,
  `codename` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_permission_content_type_id_codename_01ab375a_uniq` (`content_type_id`,`codename`),
  KEY `auth_permission_content_type_id_2f476e4b` (`content_type_id`)
) ENGINE=MyISAM AUTO_INCREMENT=41 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `auth_permission`
--

INSERT INTO `auth_permission` (`id`, `name`, `content_type_id`, `codename`) VALUES
(1, 'Can add log entry', 1, 'add_logentry'),
(2, 'Can change log entry', 1, 'change_logentry'),
(3, 'Can delete log entry', 1, 'delete_logentry'),
(4, 'Can view log entry', 1, 'view_logentry'),
(5, 'Can add permission', 2, 'add_permission'),
(6, 'Can change permission', 2, 'change_permission'),
(7, 'Can delete permission', 2, 'delete_permission'),
(8, 'Can view permission', 2, 'view_permission'),
(9, 'Can add group', 3, 'add_group'),
(10, 'Can change group', 3, 'change_group'),
(11, 'Can delete group', 3, 'delete_group'),
(12, 'Can view group', 3, 'view_group'),
(13, 'Can add user', 4, 'add_user'),
(14, 'Can change user', 4, 'change_user'),
(15, 'Can delete user', 4, 'delete_user'),
(16, 'Can view user', 4, 'view_user'),
(17, 'Can add content type', 5, 'add_contenttype'),
(18, 'Can change content type', 5, 'change_contenttype'),
(19, 'Can delete content type', 5, 'delete_contenttype'),
(20, 'Can view content type', 5, 'view_contenttype'),
(21, 'Can add session', 6, 'add_session'),
(22, 'Can change session', 6, 'change_session'),
(23, 'Can delete session', 6, 'delete_session'),
(24, 'Can view session', 6, 'view_session'),
(25, 'Can add dataset', 7, 'add_dataset'),
(26, 'Can change dataset', 7, 'change_dataset'),
(27, 'Can delete dataset', 7, 'delete_dataset'),
(28, 'Can view dataset', 7, 'view_dataset'),
(29, 'Can add user model', 8, 'add_usermodel'),
(30, 'Can change user model', 8, 'change_usermodel'),
(31, 'Can delete user model', 8, 'delete_usermodel'),
(32, 'Can view user model', 8, 'view_usermodel'),
(33, 'Can add pred model', 9, 'add_predmodel'),
(34, 'Can change pred model', 9, 'change_predmodel'),
(35, 'Can delete pred model', 9, 'delete_predmodel'),
(36, 'Can view pred model', 9, 'view_predmodel'),
(37, 'Can add testing model', 10, 'add_testingmodel'),
(38, 'Can change testing model', 10, 'change_testingmodel'),
(39, 'Can delete testing model', 10, 'delete_testingmodel'),
(40, 'Can view testing model', 10, 'view_testingmodel');

-- --------------------------------------------------------

--
-- Table structure for table `auth_user`
--

DROP TABLE IF EXISTS `auth_user`;
CREATE TABLE IF NOT EXISTS `auth_user` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `password` varchar(128) NOT NULL,
  `last_login` datetime(6) DEFAULT NULL,
  `is_superuser` tinyint(1) NOT NULL,
  `username` varchar(150) NOT NULL,
  `first_name` varchar(150) NOT NULL,
  `last_name` varchar(150) NOT NULL,
  `email` varchar(254) NOT NULL,
  `is_staff` tinyint(1) NOT NULL,
  `is_active` tinyint(1) NOT NULL,
  `date_joined` datetime(6) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `auth_user_groups`
--

DROP TABLE IF EXISTS `auth_user_groups`;
CREATE TABLE IF NOT EXISTS `auth_user_groups` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `group_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_user_groups_user_id_group_id_94350c0c_uniq` (`user_id`,`group_id`),
  KEY `auth_user_groups_user_id_6a12ed8b` (`user_id`),
  KEY `auth_user_groups_group_id_97559544` (`group_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `auth_user_user_permissions`
--

DROP TABLE IF EXISTS `auth_user_user_permissions`;
CREATE TABLE IF NOT EXISTS `auth_user_user_permissions` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `permission_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_user_user_permissions_user_id_permission_id_14a6b632_uniq` (`user_id`,`permission_id`),
  KEY `auth_user_user_permissions_user_id_a95ead1b` (`user_id`),
  KEY `auth_user_user_permissions_permission_id_1fbb5f2c` (`permission_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `dataset`
--

DROP TABLE IF EXISTS `dataset`;
CREATE TABLE IF NOT EXISTS `dataset` (
  `data_id` int(11) NOT NULL AUTO_INCREMENT,
  `data_set` varchar(100) NOT NULL,
  `dt_Accuracy` double DEFAULT NULL,
  `lr_Accuracy` double DEFAULT NULL,
  `rf_Accuracy` double DEFAULT NULL,
  `knn_Accuracy` double DEFAULT NULL,
  `dt_algo` varchar(50) DEFAULT NULL,
  `knn_algo` varchar(50) DEFAULT NULL,
  `lr_algo` varchar(50) DEFAULT NULL,
  `rf_algo` varchar(50) DEFAULT NULL,
  `svr_Accuracy` double DEFAULT NULL,
  `svr_algo` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`data_id`)
) ENGINE=MyISAM AUTO_INCREMENT=6 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `dataset`
--

INSERT INTO `dataset` (`data_id`, `data_set`, `dt_Accuracy`, `lr_Accuracy`, `rf_Accuracy`, `knn_Accuracy`, `dt_algo`, `knn_algo`, `lr_algo`, `rf_algo`, `svr_Accuracy`, `svr_algo`) VALUES
(1, 'files/data3_Pb1X5CM.csv', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
(2, 'files/finaldata1_Zp3HDas.csv', 0.00011963689866778537, NULL, NULL, NULL, 'SVR', NULL, NULL, NULL, NULL, NULL),
(3, 'files/finaldata1_Vk1sxQt.csv', 0.7302817667478403, NULL, NULL, NULL, 'DecisionTree', NULL, NULL, NULL, NULL, NULL),
(4, 'files/finaldata1_Jx9reCz.csv', 0.7091382203607587, 0.24261787271160473, NULL, NULL, 'DecisionTree', NULL, 'Linear Regressor', NULL, NULL, NULL),
(5, 'files/finaldata1_XQNlvYD.csv', 0.7136358797508446, 0.24261787271160473, 0.8039094084223285, 0.5706589548217005, 'DecisionTree', 'KNNeighbor', 'Linear Regressor', 'Random Forest', 0.00011963689866778537, 'SVR');

-- --------------------------------------------------------

--
-- Table structure for table `datatestingmodel`
--

DROP TABLE IF EXISTS `datatestingmodel`;
CREATE TABLE IF NOT EXISTS `datatestingmodel` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `Total_Stops` int(11) NOT NULL,
  `Air_India` int(11) NOT NULL,
  `GoAir` int(11) NOT NULL,
  `IndiGo` int(11) NOT NULL,
  `Jet_Airways` int(11) NOT NULL,
  `Jet_Airways_Business` int(11) NOT NULL,
  `Multiple_carriers` int(11) NOT NULL,
  `Multiple_carriers_Premium_economy` int(11) NOT NULL,
  `SpiceJet` int(11) NOT NULL,
  `Trujet` int(11) NOT NULL,
  `Vistara` int(11) NOT NULL,
  `Vistara_Premium_economy` int(11) NOT NULL,
  `Chennai` int(11) NOT NULL,
  `Delhi` int(11) NOT NULL,
  `Kolkata` int(11) NOT NULL,
  `Mumbai` int(11) NOT NULL,
  `Cochin` int(11) NOT NULL,
  `Hyderabad` int(11) NOT NULL,
  `journey_day` int(11) NOT NULL,
  `journey_month` int(11) NOT NULL,
  `Dep_Time_hour` int(11) NOT NULL,
  `Dep_Time_min` int(11) NOT NULL,
  `Arrival_Time_hour` int(11) NOT NULL,
  `Arrival_Time_min` int(11) NOT NULL,
  `dur_hour` int(11) NOT NULL,
  `dur_min` int(11) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=5 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `datatestingmodel`
--

INSERT INTO `datatestingmodel` (`id`, `Total_Stops`, `Air_India`, `GoAir`, `IndiGo`, `Jet_Airways`, `Jet_Airways_Business`, `Multiple_carriers`, `Multiple_carriers_Premium_economy`, `SpiceJet`, `Trujet`, `Vistara`, `Vistara_Premium_economy`, `Chennai`, `Delhi`, `Kolkata`, `Mumbai`, `Cochin`, `Hyderabad`, `journey_day`, `journey_month`, `Dep_Time_hour`, `Dep_Time_min`, `Arrival_Time_hour`, `Arrival_Time_min`, `dur_hour`, `dur_min`) VALUES
(1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 31, 3, 16, 55, 16, 55, 0, 0),
(2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 29, 3, 16, 55, 16, 55, 0, 0),
(3, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 27, 3, 16, 56, 16, 56, 0, 0),
(4, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 6, 4, 16, 56, 16, 56, 0, 0);

-- --------------------------------------------------------

--
-- Table structure for table `django_admin_log`
--

DROP TABLE IF EXISTS `django_admin_log`;
CREATE TABLE IF NOT EXISTS `django_admin_log` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `action_time` datetime(6) NOT NULL,
  `object_id` longtext,
  `object_repr` varchar(200) NOT NULL,
  `action_flag` smallint(5) UNSIGNED NOT NULL,
  `change_message` longtext NOT NULL,
  `content_type_id` int(11) DEFAULT NULL,
  `user_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `django_admin_log_content_type_id_c4bce8eb` (`content_type_id`),
  KEY `django_admin_log_user_id_c564eba6` (`user_id`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

-- --------------------------------------------------------

--
-- Table structure for table `django_content_type`
--

DROP TABLE IF EXISTS `django_content_type`;
CREATE TABLE IF NOT EXISTS `django_content_type` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `app_label` varchar(100) NOT NULL,
  `model` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `django_content_type_app_label_model_76bd3d3b_uniq` (`app_label`,`model`)
) ENGINE=MyISAM AUTO_INCREMENT=11 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `django_content_type`
--

INSERT INTO `django_content_type` (`id`, `app_label`, `model`) VALUES
(1, 'admin', 'logentry'),
(2, 'auth', 'permission'),
(3, 'auth', 'group'),
(4, 'auth', 'user'),
(5, 'contenttypes', 'contenttype'),
(6, 'sessions', 'session'),
(7, 'adminapp', 'dataset'),
(8, 'mainapp', 'usermodel'),
(9, 'userapp', 'predmodel'),
(10, 'userapp', 'testingmodel');

-- --------------------------------------------------------

--
-- Table structure for table `django_migrations`
--

DROP TABLE IF EXISTS `django_migrations`;
CREATE TABLE IF NOT EXISTS `django_migrations` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `app` varchar(255) NOT NULL,
  `name` varchar(255) NOT NULL,
  `applied` datetime(6) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=27 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `django_migrations`
--

INSERT INTO `django_migrations` (`id`, `app`, `name`, `applied`) VALUES
(1, 'contenttypes', '0001_initial', '2023-04-03 11:17:27.905394'),
(2, 'auth', '0001_initial', '2023-04-03 11:17:30.924895'),
(3, 'admin', '0001_initial', '2023-04-03 11:17:31.739023'),
(4, 'admin', '0002_logentry_remove_auto_add', '2023-04-03 11:17:31.743460'),
(5, 'admin', '0003_logentry_add_action_flag_choices', '2023-04-03 11:17:31.751100'),
(6, 'adminapp', '0001_initial', '2023-04-03 11:17:31.903469'),
(7, 'contenttypes', '0002_remove_content_type_name', '2023-04-03 11:17:32.211245'),
(8, 'auth', '0002_alter_permission_name_max_length', '2023-04-03 11:17:32.337615'),
(9, 'auth', '0003_alter_user_email_max_length', '2023-04-03 11:17:32.503092'),
(10, 'auth', '0004_alter_user_username_opts', '2023-04-03 11:17:32.517866'),
(11, 'auth', '0005_alter_user_last_login_null', '2023-04-03 11:17:32.653067'),
(12, 'auth', '0006_require_contenttypes_0002', '2023-04-03 11:17:32.660752'),
(13, 'auth', '0007_alter_validators_add_error_messages', '2023-04-03 11:17:32.718055'),
(14, 'auth', '0008_alter_user_username_max_length', '2023-04-03 11:17:33.155495'),
(15, 'auth', '0009_alter_user_last_name_max_length', '2023-04-03 11:17:33.318911'),
(16, 'auth', '0010_alter_group_name_max_length', '2023-04-03 11:17:33.484996'),
(17, 'auth', '0011_update_proxy_permissions', '2023-04-03 11:17:33.499542'),
(18, 'auth', '0012_alter_user_first_name_max_length', '2023-04-03 11:17:33.651743'),
(19, 'mainapp', '0001_initial', '2023-04-03 11:17:33.802345'),
(20, 'mainapp', '0002_alter_usermodel_user_image', '2023-04-03 11:17:33.818530'),
(21, 'mainapp', '0003_alter_usermodel_user_image', '2023-04-03 11:17:33.822533'),
(22, 'mainapp', '0004_alter_usermodel_user_image', '2023-04-03 11:17:33.826549'),
(23, 'sessions', '0001_initial', '2023-04-03 11:17:34.188330'),
(24, 'userapp', '0001_initial', '2023-04-03 11:17:34.373745'),
(25, 'userapp', '0002_testingmodel', '2023-04-03 11:17:34.533071'),
(26, 'adminapp', '0002_rename_accuracy_dataset_dt_accuracy_and_more', '2023-04-03 11:36:10.562478');

-- --------------------------------------------------------

--
-- Table structure for table `django_session`
--

DROP TABLE IF EXISTS `django_session`;
CREATE TABLE IF NOT EXISTS `django_session` (
  `session_key` varchar(40) NOT NULL,
  `session_data` longtext NOT NULL,
  `expire_date` datetime(6) NOT NULL,
  PRIMARY KEY (`session_key`),
  KEY `django_session_expire_date_a5c62663` (`expire_date`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

--
-- Dumping data for table `django_session`
--

INSERT INTO `django_session` (`session_key`, `session_data`, `expire_date`) VALUES
('ndxpr9jr29ljr347tmd6kqdq5rvc0953', 'eyJ1c2VyX2lkIjoxfQ:1pjIF0:7gDB0b6w7O6AZagaHaXmmV7IqJx9gh1qXdrFFYrgO9s', '2023-04-17 11:20:50.526144');

-- --------------------------------------------------------

--
-- Table structure for table `predmodel`
--

DROP TABLE IF EXISTS `predmodel`;
CREATE TABLE IF NOT EXISTS `predmodel` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `source` varchar(100) NOT NULL,
  `to` varchar(100) NOT NULL,
  `airline` varchar(100) NOT NULL,
  `dept_time` datetime(6) NOT NULL,
  `stops` int(11) NOT NULL,
  `arr_time` datetime(6) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=MyISAM AUTO_INCREMENT=5 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `predmodel`
--

INSERT INTO `predmodel` (`id`, `source`, `to`, `airline`, `dept_time`, `stops`, `arr_time`) VALUES
(1, 'Mumbai', 'Hyderabad', 'Jet_Airways', '2023-03-31 16:55:00.000000', 2, '2023-04-22 16:55:00.000000'),
(2, 'Hyderabad', 'Mumbai', 'Jet_Airways_Business', '2023-03-29 16:55:00.000000', 1, '2023-04-06 16:55:00.000000'),
(3, 'Kolkata', 'Hyderabad', 'Jet_Airways', '2023-03-27 16:56:00.000000', 2, '2023-04-21 16:56:00.000000'),
(4, 'Chennai', 'Cochin', 'IndiGo', '2023-04-06 16:56:00.000000', 2, '2023-04-29 16:56:00.000000');

-- --------------------------------------------------------

--
-- Table structure for table `user_details`
--

DROP TABLE IF EXISTS `user_details`;
CREATE TABLE IF NOT EXISTS `user_details` (
  `user_id` int(11) NOT NULL AUTO_INCREMENT,
  `user_username` varchar(100) NOT NULL,
  `user_passportnumber` varchar(20) NOT NULL,
  `user_email` varchar(100) NOT NULL,
  `user_password` varchar(100) NOT NULL,
  `user_contact` varchar(15) NOT NULL,
  `user_address` longtext NOT NULL,
  `user_image` varchar(100) NOT NULL,
  PRIMARY KEY (`user_id`)
) ENGINE=MyISAM AUTO_INCREMENT=2 DEFAULT CHARSET=latin1;

--
-- Dumping data for table `user_details`
--

INSERT INTO `user_details` (`user_id`, `user_username`, `user_passportnumber`, `user_email`, `user_password`, `user_contact`, `user_address`, `user_image`) VALUES
(1, 'user1', '961829448', 'user1@email.com', '0000', '898989898', 'hy', 'user/images/wallpaperflare.com_wallpaper_5.jpg');
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
